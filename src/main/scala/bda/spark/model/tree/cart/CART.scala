package bda.spark.model.tree.cart

import bda.common.obj.LabeledPoint
import bda.common.Logging
import bda.spark.model.tree.cart.impurity.{ImpurityAggregator, Impurity, Impurities}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * External interface of CART(Classification And Regression Trees) on spark.
  */
object CART {

  /**
    * An adapter for training a CART model.
    *
    * @param train_data training data points
    * @param impurity impurity type with [[String]], default is "Variance"
    * @param max_depth maximum depth of the CART default is 10
    * @param max_bins maximum number of bins, default is 32
    * @param bin_samples minimum number of samples used to find [[CARTSplit]] and [[CARTBin]], default is 10000
    * @param min_node_size minimum number of instances in leaves, default is 15
    * @param min_info_gain minimum infomation gain while splitting, default is 1e-6
    * @param row_rate sample ratio of training data points
    * @param col_rate sample ratio of features
    * @return an instance of [[CART]]
    */
  def train(train_data: RDD[LabeledPoint],
            impurity: String = "Variance",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            row_rate: Double = 1.0,
            col_rate: Double = 1.0): CARTModel = {

    new CART(Impurities.fromString(impurity),
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate).train(train_data)
  }

  /**
    * Method to find ID of splitting node which contains specified data point.
    *
    * @param p specified data point
    * @param root root of the CART
    * @param bins a two dimension array stored bins of all features
    * @return ID of splitting node which contains specified data point
    */
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

  /**
    * Push the child into the queue as a new splitting node if isn't [[None]].
    *
    * @param que queue stored splitting nodes
    * @param node left or right child of the splitting node
    */
  def inQueue(que :mutable.Queue[CARTNode], node: Option[CARTNode]): Unit = {
    node match {
      case Some(n) => que.enqueue(n)
      case None => // RETURN
    }
  }
}

/**
  * Class of CART(Classification And Regression Trees)
  *
  * @param impurity an instance of [[Impurity]]
  * @param max_depth maximum depth of CART
  * @param max_bins maximum number of bins
  * @param bin_samples minimum number of samples used to find [[CARTSplit]] and [[CARTBin]]
  * @param min_node_size minimum number of instances in leaves
  * @param min_info_gain minimum information gain while splitting
  * @param row_rate sample ratio of training data set
  * @param col_rate sample ratio of features
  */
class CART(impurity: Impurity,
           max_depth: Int,
           max_bins: Int,
           bin_samples: Int,
           min_node_size: Int,
           min_info_gain: Double,
           row_rate: Double,
           col_rate: Double) extends Logging {

  /**
    * Method to train a CART model over training data set.
    *
    * @param train_data training data set represented as a RDD of [[LabeledPoint]]
    * @return an instance of [[CARTModel]]
    */
  def train(train_data: RDD[LabeledPoint]): CARTModel = {
    val impurity = this.impurity
    val n_train = train_data.count().toInt
    val n_fs = train_data.map(_.fs.maxActiveIndex).max + 1
    val n_sub_fs = (n_fs * col_rate).ceil.toInt
    val n_bins = Array.fill(n_fs)(max_bins)
    // Find splits and bins for each feature
    val (splits, bins) = findSplitsBins(train_data, n_train, n_fs, n_bins)

    // Convert LabeledPoint to CARTPoint
    val cart_ps = CARTPoint.toCARTPoint(train_data, splits, n_fs, row_rate).persist()

    // Compute impurity and prediction for root node
    val root_stat = impurity.stat(cart_ps)
    val root_iprt = impurity.calculate(root_stat)
    val root_pred = impurity.predict(root_stat)
    val root = new CARTNode(1, 0, n_fs, col_rate, root_iprt, root_pred)

    val node_que = mutable.Queue(root)
    while (node_que.nonEmpty) {
      // Get splitting nodes
      val leaves = findCARTNodesToSplit(node_que)
      // Get node-id of splitting nodes
      val id_leaves = leaves.map(_.id)
      // Map node-id to position-id
      val id_pos_leaves = id_leaves.zipWithIndex.toMap
      val n_leaves = leaves.length
      val agg_leaves = cart_ps.mapPartitions {
        ps =>
          val aggs = Array.tabulate(n_leaves)(id => impurity.agg(root_stat, n_bins))
          ps.foreach {
            p =>
              val id_leaf = CART.findLeafID(p, root, bins)
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

    cart_ps.unpersist()

    new CARTModel(root,
      n_fs,
      impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate)
  }

  /**
    * Method to find best splits for splitting nodes.
    *
    * @param agg_leaves impurity aggregator for splitting nodes
    * @param n_bins number of bins for all features
    * @param n_sub_fs number of sub features
    * @param leaves an array stored leaves
    * @param bins bins of all features
    * @param impurity an instance of [[Impurity]]
    * @return (position-id, [[CARTBestSplit]])
    */
  def findBestSplits(agg_leaves: RDD[(Int, ImpurityAggregator)],
                     n_bins: Array[Int],
                     n_sub_fs: Int,
                     leaves: Array[CARTNode],
                     bins: Array[Array[CARTBin]],
                     impurity: Impurity): Map[Int, CARTBestSplit] = {
    agg_leaves.map {
      case (pos, agg) =>
        agg.toPrefixSum

        val best_split = Range(0, n_sub_fs).flatMap {
          id_sub_f =>
            val id_f = leaves(pos).sub_fs(id_sub_f)
            val n_split = n_bins(id_f) - 1
            Range(0, n_split).map {
              id_s =>
                val (l_impurity, l_predict, l_count) = agg.calLeftInfo(id_f, id_s)
                val (r_impurity, r_predict, r_count) = agg.calRightInfo(id_f, id_s)

                val weighted_impurity = impurity.calculate_weighted(l_count, r_count, l_impurity, r_impurity)

                CARTBestSplit(weighted_impurity,
                  l_impurity,
                  r_impurity,
                  l_predict,
                  r_predict,
                  l_count,
                  r_count,
                  bins(id_f)(id_s).high_split)
            }//.minBy(_.weight_impurity)
        }.minBy(_.weight_impurity)

        (pos, best_split)
    }.collectAsMap().toMap
  }

  /**
    * Method to collect splitting nodes.
    *
    * @param node_que a queue stored all splitting nodes
    * @return an array of [[CARTNode]] which will split next time
    */
  def findCARTNodesToSplit(node_que: mutable.Queue[CARTNode]): Array[CARTNode] = {
    val nodes_builder = mutable.ArrayBuilder.make[CARTNode]
    while (node_que.nonEmpty) {
      nodes_builder += node_que.dequeue()
    }
    nodes_builder.result()
  }

  /**
    * Method to find splits and bins for features.
    *
    * @param train_data training data set represented as a RDD of [[LabeledPoint]]
    * @param n_train size of training data set
    * @param n_fs number of features
    * @param n_bins number of bins for features
    * @return (Array(Array([[CARTSplit]]), Array(Array([[CARTBin]])))
    */
  def findSplitsBins(train_data: RDD[LabeledPoint],
                     n_train: Int,
                     n_fs: Int,
                     n_bins: Array[Int]): (Array[Array[CARTSplit]], Array[Array[CARTBin]]) = {

    // Sample the input data to generate splits and bins
    val n_samples = math.max(max_bins * max_bins, bin_samples)
    val r_samples = math.min(n_samples, n_train).toDouble / n_train.toDouble
    val sampled_data = train_data.sample(withReplacement = false, fraction = r_samples).collect()

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
        if (n_bin == 1) {
          bins(id_f)(0) = new CARTBin(CARTSplit.lowest(id_f), CARTSplit.highest(id_f))
        } else {
          bins(id_f)(0) = new CARTBin(CARTSplit.lowest(id_f), splits(id_f).head)
          Range(1, n_split).foreach {
            id_bin =>
              bins(id_f)(id_bin) = new CARTBin(splits(id_f)(id_bin - 1), splits(id_f)(id_bin))
          }
          bins(id_f)(n_split) = new CARTBin(splits(id_f).last, CARTSplit.highest(id_f))
        }
    }

    (splits, bins)
  }

  /**
    * Method to find splits for specified feature with sampled training data set.
    *
    * @param sampled_f sampled value of specified feature
    * @param id_f ID of specified feature, indexed from 0
    * @param n_bins nube of bins for all features
    * @return an array stored values of splits for specified feature
    */
  def findSplitVS(sampled_f: Array[Double],
                  id_f: Int,
                  n_bins: Array[Int]): Array[Double] = {

    // Count number of each distinct value
    val cnt = sampled_f.foldLeft(Map.empty[Double, Int]) {
      (m, v) => m + ((v, m.getOrElse(v, 0) + 1))
    }.toArray.sortBy(_._1)

    val possible_vs = cnt.length
    if (possible_vs <= n_bins(id_f)) {
      cnt.map(_._1).slice(1, possible_vs)
    } else {
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