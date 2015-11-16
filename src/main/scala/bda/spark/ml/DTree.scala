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
    val dt_points = DTreePoint.convertToTreeRDD(input, bins, metadata).persist()
    Log.log("INFO", "Convert to DTreePoint done.")

    // create root for decision tree
    val root = Node.empty(id = 1, 0)

    // create a node queue which help to generate a binary tree
    val node_que = new mutable.Queue[Node]()
    node_que.enqueue(root)

    while (node_que.nonEmpty) {
      val leaves = DTree.findSplittingNodes(node_que)

      val num_leaves = leaves.length
      val id_leaves = leaves.map { case node =>
        node.id
      }

      val stats_aggregator = dt_points.mapPartitions { points =>
        // create Stats for each partition
        val stats = id_leaves.map { id =>
          val node_stats = new Array[Array[Stat]](metadata.num_features)
          for (i <- 0 until metadata.num_features) {
            val stat_builder = mutable.ArrayBuilder.make[Stat]
            for (j <- 0 until metadata.num_bins(i)) {
              stat_builder += Stat.empty
            }
            node_stats(i) = stat_builder.result()
          }
          (id, node_stats)
        }.toMap

        // statistic sum, squared sum and count for each bin
        points.foreach { p =>
          // find leave id which is corresponding with this point
          val id_leaf = DTree.findLeafId(p, root)
          // update stats
          if (id_leaves.contains(id_leaf)) {
            for (index_feature <- 0 until metadata.num_features) {
              stats(id_leaf)(index_feature)(p.binned_features(index_feature)).update(p.label)
            }
          }
        }

        stats.toArray.iterator
      }

      val stats = stats_aggregator.reduceByKey { (stats_a, stats_b) =>
        for (index_feature <- 0 until metadata.num_features) {
          for (index_bin <- 0 until metadata.num_bins(index_feature)) {
            stats_a(index_feature)(index_bin).merge(stats_b(index_feature)(index_bin))
          }
        }
        stats_a
      }.collectAsMap()

      // find best splits for leaves
      leaves.foreach { node =>
        // find best split for node
        DTree.findBestSplit(node, stats(node.id), metadata, dt_para, bins)
        // split this node
        DTree.split(node)
        // push left child and right child of this node into queue
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

      // get count for each distinct value
      val value_cnts = sampled_features.foldLeft(Map.empty[Double, Int]) { (m, x) =>
        m + ((x, m.getOrElse(x, 0) + 1))
      }.toSeq.sortBy(_._1).toArray

      // if possible splits is not enough or just enough, just return all possible splits
      val possible_splits = value_cnts.length
      if (possible_splits <= num_splits) {
        value_cnts.map(_._1)
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

  def findLeafId(point: DTreePoint, root: Node): Int = {
    var leaf = root
    while (!leaf.is_leaf) {
      if (point.features(leaf.split.get.feature) <= leaf.split.get.threshold) {
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

  def split(node: Node): Unit = {
    node.split match {
      case Some(split) =>
        node.is_leaf = false
        node.left_child = Some(Node.generate_lchild(node))
        node.right_child = Some(Node.generate_rchild(node))
      case _ =>
    }
  }

  def inQueue(que: mutable.Queue[Node], node: Option[Node]): Unit = {
    node match {
      case Some(n)  => que.enqueue(n)
      case _ =>
    }
  }
}