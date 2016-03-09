package bda.spark.model.tree.cart

import bda.common.obj.LabeledPoint
import bda.common.Logging
import bda.spark.model.tree.cart.impurity.{ImpurityAggregator, Impurity, Impurities}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object CART {

  def train(train_data: RDD[LabeledPoint],
            impurity: String = "Variance",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            row_rate: Double = 1.0,
            col_rate: Double = 1.0): Unit = {

    new CART(Impurities.fromString(impurity),
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate).train(train_data)
  }

  def findLeafID(p: CARTPoint, root: CARTNode, bins: Array[Array[CARTBin]]): Int = {
    var leaf = root
    while (!leaf.is_leaf) {
      val split = leaf.split.get
      if (bins(split.id_f)(p.binned_fs(split.id_f)).high_split.threshold <= split.threshold) {
        leaf = leaf.left_child.get
      } else {
        leaf = leaf.right_child.get
      }

    }
    leaf.id
  }

  def inQueue(que :mutable.Queue[CARTNode], node: Option[CARTNode]): Unit = {
    node match {
      case Some(n) => que.enqueue(n)
      case None => // RETURN
    }
  }
}

class CART(impurity: Impurity,
           max_depth: Int,
           max_bins: Int,
           bin_samples: Int,
           min_node_size: Int,
           min_info_gain: Double,
           row_rate: Double,
           col_rate: Double) extends Logging {

  def train(train_data: RDD[LabeledPoint]) = {
    val impurity = this.impurity
    val n_train = train_data.count().toInt
    val n_fs = train_data.map(_.fs.maxActiveIndex).max + 1
    val n_sub_fs = (n_fs * col_rate).ceil.toInt
    val n_bins = Array.fill(n_fs)(max_bins)
    // Find splits and bins for each feature
    val (splits, bins) = findSplitsBins(train_data, n_train, n_fs, n_bins)

    // Convert LabeledPoint to CARTPoint
    val cart_ps = CARTPoint.toCARTPoint(train_data, splits, n_fs, row_rate)

    val root_stat = impurity.stat(cart_ps)
    val root_iprt = impurity.calculate(root_stat)
    val root_pred = impurity.predict(root_stat)
    val root = new CARTNode(1, 0, n_fs, col_rate, root_iprt, root_pred)
    println(s"root_stat($root_stat), root_iprt($root_iprt), root_pred($root_pred)")

    val node_que = mutable.Queue(root)
    while (node_que.nonEmpty) {
      val leaves = findCARTNodesToSplit(node_que)
      println(s"size(leaves)=${leaves.length}")
      val id_leaves = leaves.map(_.id)
      val id_pos_leaves = id_leaves.zipWithIndex.toMap
      val n_leaves = leaves.length
      val agg_leaves = cart_ps.mapPartitions {
        ps =>
          val aggs = Array.tabulate(n_leaves)(id => impurity.agg(root_stat, n_bins))

          ps.foreach {
            p =>
              val id_leaf = CART.findLeafID(p, root, bins)
              println(s"p($p), id_leaf($id_leaf)")
              if (id_pos_leaves.contains(id_leaf)) {
                val pos_leaf = id_pos_leaves(id_leaf)
                aggs(pos_leaf).update(p, leaves(pos_leaf).sub_fs)
              }
          }
          aggs.view.zipWithIndex.map(_.swap).iterator
      }.reduceByKey((a, b) => a.merge(b))

      // Find best splits for leaves
      val best_splits = findBestSplits(agg_leaves, n_bins, n_sub_fs, leaves, bins, impurity)

      best_splits.foreach {
        case (pos, best_split) =>
          leaves(pos).split(best_split, max_depth, min_info_gain, min_node_size)
          CART.inQueue(node_que, leaves(pos).left_child)
          CART.inQueue(node_que, leaves(pos).right_child)
      }

    }

    CARTModel.printStructure(root)
  }

  def findBestSplits(agg_leaves: RDD[(Int, ImpurityAggregator)],
                     n_bins: Array[Int],
                     n_sub_fs: Int,
                     leaves: Array[CARTNode],
                     bins: Array[Array[CARTBin]],
                     impurity: Impurity): Map[Int, CARTBestSplit] = {
    agg_leaves.map {
      case (pos, agg) =>
        agg.toPrefixSum(n_bins)

        val best_split = Range(0, n_sub_fs).map {
          id_sub_f =>
            val id_f = leaves(pos).sub_fs(id_sub_f)
            val n_split = n_bins(id_f) - 1
            Range(0, n_split).map {
              id_s =>
                val (l_impurity, l_predict, l_count) = agg.calLeftInfo(id_f, id_s)
                val (r_impurity, r_predict, r_count) = agg.calRightInfo(id_f, id_s)

                val weighted_impurity = impurity.calculate_weighted(l_count, r_count, l_impurity, r_impurity)

                println(s"id_f($id_f),id_b($id_s)")
                println(s"l_impurity($l_impurity),l_predict($l_predict),l_total($l_count)")
                println(s"r_impurity($r_impurity),r_predict($r_predict),r_total($r_count)")
                println(s"weighted_impurity($weighted_impurity)")

                CARTBestSplit(weighted_impurity,
                  l_impurity,
                  r_impurity,
                  l_predict,
                  r_predict,
                  l_count,
                  r_count,
                  bins(id_f)(id_s).high_split)
            }.minBy(_.weight_impurity)
        }.minBy(_.weight_impurity)

        (pos, best_split)
    }.collectAsMap().toMap
  }

  def findCARTNodesToSplit(node_que: mutable.Queue[CARTNode]): Array[CARTNode] = {
    val nodes_builder = mutable.ArrayBuilder.make[CARTNode]
    while (node_que.nonEmpty) {
      nodes_builder += node_que.dequeue()
    }
    nodes_builder.result()
  }

  def findSplitsBins(train_data: RDD[LabeledPoint],
                     n_train: Int,
                     n_fs: Int,
                     n_bins: Array[Int]) = {

    // Sample the input data to generate splits and bins
    val n_samples = math.max(max_bins * max_bins, bin_samples)
    val r_samples = math.min(n_samples, n_train).toDouble / n_train.toDouble
    val sampled_data = train_data.sample(withReplacement = false, fraction = r_samples).collect()
    logInfo(s"rate(samples)=$r_samples")

    val splits = new Array[Array[CARTSplit]](n_fs)
    val bins = new Array[Array[CARTBin]](n_fs)

    Range(0, n_fs).foreach {
      id_f =>
        val sampled_f = sampled_data.map(_.fs(id_f))
        val split_vs = findSplitVS(sampled_f, id_f, n_bins)

        val n_split = split_vs.length
        val n_bin = n_split + 1
        n_bins(id_f) = n_bin

        splits(id_f) = new Array[CARTSplit](n_split)
        bins(id_f) = new Array[CARTBin](n_bin)

        // Generate splits
        Range(0, n_split).foreach {
          id_split =>
            splits(id_f)(id_split) = new CARTSplit(id_f, split_vs(id_split))
        }
        // Generate bins
        bins(id_f)(0) = new CARTBin(CARTSplit.lowest(id_f), splits(id_f).head)
        Range(1, n_split).foreach {
          id_bin =>
            bins(id_f)(id_bin) = new CARTBin(splits(id_f)(id_bin - 1), splits(id_f)(id_bin))
        }
        bins(id_f)(n_split) = new CARTBin(splits(id_f).last, CARTSplit.highest(id_f))

        println(s"HouJP >> id_f($id_f), split values(${split_vs.mkString(",")})")
    }

    (splits, bins)
  }

  def findSplitVS(sampled_f: Array[Double], id_f: Int, n_bins: Array[Int]): Array[Double] = {

    // Count number of each distinct value
    val cnt = sampled_f.foldLeft(Map.empty[Double, Int]) {
      (m, v) => m + ((v, m.getOrElse(v, 0) + 1))
    }.toArray.sortBy(_._1)

    val possible_vs = cnt.length
    if (possible_vs <= n_bins(id_f)) {
      cnt.map(_._1).slice(1, possible_vs)
    } else {
      val svs_builder = mutable.ArrayBuilder.make[Double]
      val ave = sampled_f.length.toDouble / n_bins(id_f)
      val n_sp = n_bins(id_f) - 1
      val sp = new Array[Double](n_sp)

      var acc_cnt = cnt(0)._2
      var acc_ave = ave
      var now_bias = math.abs(acc_ave - acc_cnt)

      var id = 1
      var id_sp = 0
      while (id < cnt.length) {
        if ((n_sp - id_sp) < (cnt.length - id)) {
          val pre_bias = now_bias
          acc_cnt += cnt(id)._2
          now_bias = math.abs(acc_ave - acc_cnt)

          if (pre_bias < now_bias) {
            sp(id_sp) = cnt(id)._1
            acc_ave += ave
            now_bias = math.abs(acc_ave - acc_cnt)
            id_sp += 1
          }
        } else {
          sp(id_sp) = cnt(id)._1
          id_sp += 1
        }
        id += 1
      }
      sp
    }
  }
}