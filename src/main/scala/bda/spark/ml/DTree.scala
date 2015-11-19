package bda.spark.ml

import bda.local.ml.model.LabeledPoint
import bda.local.ml.util.Log
import bda.spark.ml.model._
import bda.spark.ml.para.DTreePara
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.util.Random

class DTree(private val dt_para: DTreePara) {

  def fit(input: RDD[LabeledPoint]): DTreeModel = {

    val metadata = DTreeMetaData.build(input, dt_para)

    // find splits and bins for each feature
    val (splits, bins) = DTree.findSplitsBins(input, metadata, dt_para)
    Log.log("INFO", "Find splits and bins done.")

    // convert LabeledPoint to TreePoint which used bin-index instead of feature-value
    val dt_points: RDD[DTreePoint] = DTreePoint.convertToTreeRDD(input, bins, metadata).persist()
    Log.log("INFO", "Convert to DTreePoint done.")

    // create root for decision tree
    val root = Node.empty(id = 1, 0)
    // calculate root's impurity and root's predict
    val root_count = metadata.num_examples
    val root_sum = dt_points.map(_.label).sum()
    val root_squared_sum = dt_points.map(p => p.label * p.label).sum()
    val root_stat = new Stat(root_count, root_sum, root_squared_sum)
    root.impurity = root_stat.cal_impurity(dt_para.impurity_calculator)
    root.predict = root_stat.cal_prediction(dt_para.loss_calculator)

    // create a node queue which help to generate a binary tree
    val node_que = new mutable.Queue[Node]()
    node_que.enqueue(root)

    while (node_que.nonEmpty) {
      val leaves = DTree.findSplittingNodes(node_que)

      val num_leaves = leaves.length
      val id_leaves = leaves.map { case node =>
        node.id
      }
      val ind_leaves = id_leaves.zipWithIndex.toMap

      val agg_leaves = dt_points.mapPartitions { points =>
        // create aggregators for each partition and reduce by key
        val agg_leaves = Array.tabulate(num_leaves)(index => new DTreeStatsAgg(metadata))

        points.foreach { p =>
          val id_leaf = DTree.findLeafId(p, root, bins)
          if (ind_leaves.contains(id_leaf)) {
            val ind_leaf = ind_leaves(id_leaf)
            agg_leaves(ind_leaf).update(p)
          }
        }

        agg_leaves.view.zipWithIndex.map(_.swap).iterator
      }.reduceByKey((a, b) => a.merge(b))

      val best_splits = DTree.findBestSplit(agg_leaves, dt_para, metadata)
//      Log.log("INFO", "<fit> best_splits:")
//      best_splits.foreach(println)
      DTree.updateBestSplit(leaves, best_splits, bins, metadata, dt_para)

//      val stats_aggregator = dt_points.mapPartitions { points =>
//
//        // create Stats for each partition
//        val stats = id_leaves.map { id =>
//          val node_stats = new Array[Array[Stat]](metadata.num_features)
//          for (i <- 0 until metadata.num_features) {
//            val stat_builder = mutable.ArrayBuilder.make[Stat]
//            for (j <- 0 until metadata.num_bins(i)) {
//              stat_builder += Stat.empty
//            }
//            node_stats(i) = stat_builder.result()
//          }
//          (id, node_stats)
//        }.toMap
//
//        // statistic sum, squared sum and count for each bin
//        points.foreach { p =>
//          // find leave id which is corresponding with this point
//          val id_leaf = DTree.findLeafId(p, root)
//          // update stats
//          if (id_leaves.contains(id_leaf)) {
//            for (index_feature <- 0 until metadata.num_features) {
//              stats(id_leaf)(index_feature)(p.binned_features(index_feature)).update(p.label)
//            }
//          }
//        }
//
//        stats.toArray.iterator
//      }
//
//      val stats: RDD[(Int, Array[Array[Stat]])] = stats_aggregator.reduceByKey { (stats_a, stats_b) =>
//        for (index_feature <- 0 until metadata.num_features) {
//          for (index_bin <- 0 until metadata.num_bins(index_feature)) {
//            stats_a(index_feature)(index_bin).merge(stats_b(index_feature)(index_bin))
//          }
//        }
//        stats_a
//      }

      //val stats_map = stats.collectAsMap()

      // find best_splits for leaves by distributed operation
//      val nodes_best_split = DTree.findBestSplit(stats, metadata, dt_para)
//      Log.log("INFO", "<fit> best_splits:")
//      nodes_best_split.foreach(println)
//      DTree.updateBestSplit(leaves, nodes_best_split, bins, metadata, dt_para)

      leaves.foreach { node =>
        DTree.inQueue(node_que, node.left_child)
        DTree.inQueue(node_que, node.right_child)
      }
    }
    dt_points.unpersist()

    new DTreeModel(root, dt_para)
  }
}

object DTree {

  def dfs(node: Node): Unit = {

    node.left_child match {
      case Some(n) => dfs(n)
      case _ =>
    }

    node.right_child match {
      case Some(n) => dfs(n)
      case _ =>
    }
  }

  def findSplitsBins(input: RDD[LabeledPoint], metadata: DTreeMetaData, dt_para: DTreePara): (Array[Array[Split]], Array[Array[Bin]]) = {
    val num_features = metadata.num_features

    // sample the input
    val required_samples = math.max(metadata.max_bins * metadata.max_bins, dt_para.min_samples)
    val fraction = if (required_samples < metadata.num_examples) {
      required_samples.toDouble / metadata.num_examples.toDouble
    } else {
      1.0
    }
    val sampled_input = input.sample(withReplacement = false, fraction, new Random().nextLong()).collect()

    val splits = new Array[Array[Split]](num_features)
    val bins = new Array[Array[Bin]](num_features)

    var index_feature = 0
    while (index_feature < num_features) {
      val sampled_features = sampled_input.map(lp => lp.features(index_feature))
      val feature_splits = findSplits(sampled_features, metadata, index_feature)

      val num_splits = feature_splits.length
      val num_bins = num_splits + 1

      splits(index_feature) = new Array[Split](num_splits)
      bins(index_feature) = new Array[Bin](num_bins)

      for (index_split <- 0 until feature_splits.length) {
        splits(index_feature)(index_split) = new Split(index_feature, feature_splits(index_split))
      }

      bins(index_feature)(0) = new Bin(new LowestSplit(index_feature), splits(index_feature).head)
      for (index_bin <- 1 until feature_splits.length) {
        bins(index_feature)(index_bin) = new Bin(splits(index_feature)(index_bin - 1), splits(index_feature)(index_bin))
      }
      bins(index_feature)(feature_splits.length) = new Bin(splits(index_feature).last, new HighestSplit(index_feature))

      index_feature += 1
    }

    (splits, bins)
  }

  def findSplits(
      sampled_features: Array[Double],
      metadata: DTreeMetaData,
      index_feature: Int): Array[Double] = {

    val splits = {
      val num_splits = metadata.numSplits(index_feature)
      //Log.log("INFO", s"<findSplit> index_feature = ${index_feature}, num_splits = $num_splits")

      // get count for each distinct value
      val value_cnts = sampled_features.foldLeft(Map.empty[Double, Int]) { (m, x) =>
        m + ((x, m.getOrElse(x, 0) + 1))
      }.toSeq.sortBy(_._1).toArray

      // if possible splits is not enough or just enough, just return all possible splits
      val possible_splits = value_cnts.length
      //Log.log("INFO", s"<findSplit> possible_splits = $possible_splits")
      if (possible_splits <= num_splits) {
        value_cnts.map(_._1).take(possible_splits - 1)
      } else {
        val stride: Double = sampled_features.length.toDouble / (num_splits + 1)

        val split_builder = mutable.ArrayBuilder.make[Double]
        var index = 1
        var cur_cnt = value_cnts(0)._2
        var target_cnt = stride
        while (index < value_cnts.length) {
          val pre_cnt = cur_cnt
          cur_cnt += value_cnts(index)._2
          val pre_gap = math.abs(pre_cnt - target_cnt)
          val cur_gap = math.abs(cur_cnt - target_cnt)
          if (pre_gap < cur_gap) {
            split_builder += value_cnts(index - 1)._1
            target_cnt += stride
          }
          index += 1
        }
        val tmp = split_builder.result()
        //Log.log("INFO", s"<findSplit> tmp.length = ${tmp.length}")
        split_builder.result()
      }
    }

    // the feature which has only one value is useless, you should delete it from features
    require(splits.length > 0, s"DTree could not handle feature $index_feature since it had only 1 unique value." +
      " Please remove this feature and try again.")

    // reset features' bin-number which maybe changed
    metadata.setBins(index_feature, splits.length)

    splits
  }

  def findSplittingNodes(node_que: mutable.Queue[Node]): Array[Node] = {
    val leaves_builder = mutable.ArrayBuilder.make[Node]
    while (node_que.nonEmpty) {
      leaves_builder += node_que.dequeue()
    }
    leaves_builder.result()
  }

  def findLeafId(point: DTreePoint, root: Node, bins: Array[Array[Bin]]): Int = {
    var leaf = root
    while (!leaf.is_leaf) {
      val split = leaf.split.get
      if (bins(split.feature)(point.binned_features(split.feature)).high_split.threshold <= split.threshold) {
//      if (point.features(leaf.split.get.feature) <= leaf.split.get.threshold) {
        leaf = leaf.left_child.get
      } else {
        leaf = leaf.right_child.get
      }
    }
    leaf.id
  }

  def findBestSplit(
      node: Node,
      stats: Array[Array[Stat]],
      metadata: DTreeMetaData,
      dt_para: DTreePara,
      bins: Array[Array[Bin]]): Unit = {

    val father_stat = Stat.empty
    for (index_bin <- 0 until metadata.num_bins(0)) {
      father_stat.merge(stats(0)(index_bin))
    }
    // update node's impurity and predict
    node.predict = father_stat.cal_prediction(dt_para.loss_calculator)
    node.impurity = father_stat.cal_impurity(dt_para.impurity_calculator)

    var best_split: Option[Split] = None
    var min_impurity = node.impurity

    // judge if the depth satisfied the condition
    if (node.depth < dt_para.max_depth) {
      for (index_feature <- 0 until metadata.num_features) {
        val lstat = Stat.empty
        val rstat = father_stat.copy
        for (index_bin <- 0 until metadata.num_bins(index_feature) - 1) {
          lstat.merge(stats(index_feature)(index_bin))
          rstat.disunify(stats(index_feature)(index_bin))

          val lchild_impurity = lstat.cal_impurity(dt_para.impurity_calculator)
          val rchild_impurity = rstat.cal_impurity(dt_para.impurity_calculator)
          val weighted_impurity = lchild_impurity * lstat.count / father_stat.count + rchild_impurity * rstat.count / father_stat.count
          if ((weighted_impurity < min_impurity)
            && (lstat.count >= dt_para.min_node_size)
            && (rstat.count >= dt_para.min_node_size)) {
            min_impurity = weighted_impurity
            best_split = Some(bins(index_feature)(index_bin).high_split)
          }
        }
      }
    }

    node.split = best_split
  }

  def inQueue(que: mutable.Queue[Node], node: Option[Node]): Unit = {
    node match {
      case Some(n)  => que.enqueue(n)
      case _ =>
    }
  }

  def findBestSplit(agg_leaves: RDD[(Int, DTreeStatsAgg)], dt_para: DTreePara, metadata: DTreeMetaData): scala.collection.Map[Int, (Double, Int, Int, Double, Double, Double, Double, Int, Int)] = {
    val num_fs = metadata.num_features

    agg_leaves.map { case (ind, agg) =>
      agg.toPrefixSum()

      val best_split = Range(0, metadata.num_features).map { index_f =>
        val num_splits = metadata.numSplits(index_f)
        Range(0, num_splits).map { index_b =>
          val (l_impurity, l_pred, l_cnt) = agg.calLeftInfo(index_f, index_b, dt_para.impurity_calculator, dt_para.loss_calculator)
          val (r_impurity, r_pred, r_cnt) = agg.calRightInfo(index_f, index_b, dt_para.impurity_calculator, dt_para.loss_calculator)
          val f_cnt = l_cnt + r_cnt

          val weighted_impurity = l_impurity * l_cnt / f_cnt + r_impurity * r_cnt / f_cnt

          (weighted_impurity, index_f, index_b, l_impurity, r_impurity, l_pred, r_pred, l_cnt, r_cnt)
        }.minBy(_._1)
      }.minBy(_._1)
      (ind, best_split)
    }.collectAsMap()
  }

  def findBestSplit(
      stats: RDD[(Int, Array[Array[Stat]])],
      metadata: DTreeMetaData,
      dt_para: DTreePara): scala.collection.Map[Int, (Double, Int, Int, Double, Double, Double, Double, Int, Int)] = {

    stats.map { case (id, node_stats) =>
      val best_split = Range(0, metadata.num_features).map { case index_f =>
        val num_splits = metadata.numSplits(index_f)
        val num_bins = metadata.num_bins(index_f)

        var index_b = 1
        while (index_b < num_bins) {
          node_stats(index_f)(index_b).merge(node_stats(index_f)(index_b - 1))
          index_b += 1
        }

        Range (0, num_splits).map { case index_s =>
          val l_stat = node_stats(index_f)(index_s)
          val r_stat = node_stats(index_f)(num_bins - 1).copy.disunify(l_stat)

          val l_impurity = l_stat.cal_impurity(dt_para.impurity_calculator)
          val r_impurity = r_stat.cal_impurity(dt_para.impurity_calculator)
          val weighted_impurity = Stat.cal_weighted_impurity(l_stat, r_stat, l_impurity, r_impurity)

          val l_pred = l_stat.cal_prediction(dt_para.loss_calculator)
          val r_pred = r_stat.cal_prediction(dt_para.loss_calculator)

          val l_cnt = l_stat.count
          val r_cnt = r_stat.count

          (weighted_impurity, index_f, index_s, l_impurity, r_impurity, l_pred, r_pred, l_cnt, r_cnt)
        }.minBy(_._1)
      }.minBy(_._1)
      (id, best_split)
    }.collectAsMap()
  }

  def updateBestSplit(
      leaves: Array[Node],
      best_splits: scala.collection.Map[Int, (Double, Int, Int, Double, Double, Double, Double, Int, Int)],
      bins: Array[Array[Bin]],
      metadata: DTreeMetaData,
      dt_para: DTreePara): Unit = {

    val num_leaves = leaves.length

    var index = 0
    while (index < num_leaves) {
      val node = leaves(index)
      val (weighted_impurity, split_f, split_s, l_impurity, r_impurity, l_pred, r_pred, l_cnt, r_cnt) = best_splits(index)
      val info_gain = node.impurity - weighted_impurity
      val split = bins(split_f)(split_s).high_split
      if ((weighted_impurity >= dt_para.min_info_gain)
        && (node.depth < dt_para.max_depth)
        && (l_cnt >= dt_para.min_node_size)
        && (r_cnt >= dt_para.min_node_size)) {

        node.is_leaf = false
        node.split = Some(split)
        node.generate_lchild(l_impurity, l_pred)
        node.generate_rchild(r_impurity, r_pred)
      }

      index += 1
    }
  }
}