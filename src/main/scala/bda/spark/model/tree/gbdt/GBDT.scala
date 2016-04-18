package bda.spark.model.tree.gbdt

import bda.common.Logging
import bda.common.obj.LabeledPoint
import bda.common.util.{Timer, Msg}
import bda.spark.model.tree.{TreeNode, NodeBestSplit, FeatureBin, FeatureSplit}
import bda.spark.model.tree.gbdt.impurity.{ImpurityAggregator, Impurities, Impurity}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * External interface of GBDT(Gradient Boosting Decisision Trees) on spark.
  */
object GBDT extends Logging {

  /**
    * An adapter for training a GBDT model.
    *
    * @param train_data     training data set
    * @param impurity       impurity used to split node, default is "Variance"
    * @param max_depth      maximum depth of the CART default is 10
    * @param max_bins       maximum number of bins, default is 32
    * @param bin_samples    minimum number of samples used to find [[bda.spark.model.tree.FeatureSplit]] and [[bda.spark.model.tree.FeatureBin]], default is 10000
    * @param min_node_size  minimum number of instances in leaves, default is 15
    * @param min_info_gain  minimum infomation gain while splitting, default is 1e-6
    * @param num_round      number of rounds for GBDT
    * @return an instance of [[GBDTModel]]
    */
  def train(train_data: RDD[LabeledPoint],
            impurity: String = "Variance",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            num_round: Int = 10): GBDTModel = {

    val msg = Msg("n(train_data)" -> train_data.count(),
      "impurity" -> impurity,
      "max_depth" -> max_depth,
      "max_bins" -> max_bins,
      "bin_samples" -> bin_samples,
      "min_node_size" -> min_node_size,
      "min_info_gain" -> min_info_gain,
      "num_round" -> num_round
    )
    logInfo(msg.toString)

    new GBDT(impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      num_round).train(train_data)
  }

  /**
    * Method to find ID of splitting node which contains specified data point.
    *
    * @param p specified data point
    * @param roots roots of the K CART model
    * @param bins a two dimension array stored bins of all features
    * @return ID of splitting node which contains specified data point
    */
  def findLeafID(p: CARTPoint, roots: Seq[TreeNode], bins: Array[Array[FeatureBin]]): Seq[(Int, Int)] = {
    roots.map {
      root =>
        findLeafID(p, root, bins)
    }
  }

  /**
    * Method to find ID of splitting node which contains specified data point.
    *
    * @param p specified data point
    * @param root root of the specified CART model
    * @param bins a two dimension array stored bins of all features
    * @return (node-ID, label-ID) of splitting node which contains specified data point
    */
  def findLeafID(p: CARTPoint, root: TreeNode, bins: Array[Array[FeatureBin]]): (Int, Int) = {
    var leaf = root
    while (!leaf.is_leaf) {
      val split = leaf.split.get
      if (bins(split.id_f)(p.binned_fs(split.id_f)).high_split.threshold <= split.threshold) {
        leaf = leaf.left_child.get
      } else {
        leaf = leaf.right_child.get
      }

    }
    (leaf.id, leaf.id_label)
  }

  /**
    * Push the child into the queue as a new splitting node if isn't [[None]].
    *
    * @param que queue stored splitting nodes
    * @param node left or right child of the splitting node
    */
  def inQueue(que :mutable.Queue[TreeNode], node: Option[TreeNode]): Unit = {
    node match {
      case Some(n) => que.enqueue(n)
      case None => // RETURN
    }
  }
}

/**
  * Class of GBDT(Gradient Boosting Decision Tree).
  *
  * @param impurity       impurity used to split node
  * @param max_depth      maximum depth of CART
  * @param max_bins       maximum number of bins
  * @param bin_samples    minimum number of samples used to find [[bda.spark.model.tree.FeatureSplit]] and [[FeatureBin]]
  * @param min_node_size  minimum number of instances in leaves
  * @param min_info_gain  minimum information gain while splitting
  * @param num_round      number of rounds for GBDT
  */
class GBDT(impurity: String,
           max_depth: Int,
           max_bins: Int,
           bin_samples: Int,
           min_node_size: Int,
           min_info_gain: Double,
           num_round: Int) extends Logging{

  /**
    * Method to train a GBDT model based on training data set.
    *
    * @param train_data training data set represented as a RDD of [[LabeledPoint]]
    * @return           an instance of [[GBDTModel]]
    */
  def train(train_data: RDD[LabeledPoint]): GBDTModel = {
    val timer = new Timer()

    // Statistic information about training data
    val impurity = Impurities.fromString(this.impurity)
    val n_train = train_data.count().toInt
    val n_label = train_data.map(_.label.toInt).max() + 1
    val n_feature = train_data.map(_.fs.maxActiveIndex).max + 1
    val n_bins = Array.fill(n_feature)(max_bins)

    // Build container for roots
    var wk_learners = new Array[TreeNode](0)

    // Find splits and bins fo each featue
    val (splits, bins) = findSplitsBins(train_data, n_train, n_feature, n_bins)

    // Convert LabeledPoint to GBDTPoint
    var gbdt_ps = GBDTPoint.toGBDTPoint(train_data, splits, n_label, n_feature).persist()
    // Convert GBDTPoint to CARTPoint
    val cart_ps = CARTPoint.toCARTPoint(gbdt_ps)

    // Build weak learner #0
    val wl0 = buildCART(0, cart_ps, n_train, n_label, n_feature, n_bins, bins, impurity)
    wk_learners ++= wl0

    logInfo("GBDT Model round#1 training done")

    Range(1, num_round).foreach {
      iter =>

        // Update GBDTPoint
        gbdt_ps = gbdt_ps.map {
          p =>
            val new_f_K = Range(0, n_label).map {
              id =>
                p.f_K(id) + GBDTModel.predict(p.fs, wk_learners((iter - 1) * n_label + id))
            }.toArray
            GBDTPoint(p.id, p.label, new_f_K, p.fs, p.binned_fs)
        }
        // Update CARTPoint
        val cart_ps = CARTPoint.toCARTPoint(gbdt_ps)
        // Build weak learner #iter
        val wl = buildCART(iter, cart_ps, n_train, n_label, n_feature, n_bins, bins, impurity)
        wk_learners ++= wl

        logInfo(s"GBDT Model round#${iter + 1} training done")
    }

    logInfo(s"GBDT Model training done, cost time ${timer.cost()}ms")

    new GBDTModel(impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      num_round,
      n_label,
      wk_learners)
  }

  /**
    * Method to train a CART model as a weak learner of GBDT.
    *
    * @param iter      id of iteration
    * @param cart_ps   training data set used to build CART model
    * @param n_train   number of training data set
    * @param n_label   number of different labels
    * @param n_feature number of features
    * @param n_bins    number of bins for all features
    * @param bins      bins of all features
    * @param impurity  the impurity used to split node of CART
    * @return
    */
  def buildCART(iter: Int,
                cart_ps: RDD[CARTPoint],
                n_train: Int,
                n_label: Int,
                n_feature: Int,
                n_bins: Array[Int],
                bins: Array[Array[FeatureBin]],
                impurity: Impurity): Seq[TreeNode] = {

    cart_ps.persist()

    // Compute impurity and prediction for K root nodes
    val root_stat = impurity.stat(n_label, cart_ps)
    val root_iprt = impurity.calculate(root_stat)
    val root_pred = impurity.predict(root_stat)

    // Generate roots of CART for K labels
    val root = Range(0, n_label).map {
      id =>
        new TreeNode(1, id, n_train, 0, n_feature, 1.0, root_iprt(id), root_pred(id))
    }.toSeq

    val node_que = mutable.Queue[TreeNode]()
    root.foreach(node_que.enqueue(_))

    while (node_que.nonEmpty) {
      // Get splitting nodes
      val leaves = findCARTNodesToSplit(node_que)
      // Get (node-id, label-id) of splitting nodes
      val id_leaves = leaves.map(e => (e.id, e.id_label))
      // Map (node-id, label-id) to position-id
      val id_pos_leaves = id_leaves.zipWithIndex.toMap
      val n_leaves = leaves.length
      val agg_leaves = cart_ps.mapPartitions {
        ps =>
          val aggs = Array.tabulate(n_leaves)(id => impurity.agg(n_feature, n_bins))
          ps.foreach {
            p =>
              val ids = GBDT.findLeafID(p, root, bins)
              ids.foreach {
                e =>
                  if (id_pos_leaves.contains(e)) {
                    val pos_leaf = id_pos_leaves(e)
                    aggs(pos_leaf).update(p, e._2)
                  }
              }
          }
          aggs.view.zipWithIndex.map(_.swap).iterator
      }.reduceByKey((a, b) => a.merge(b))

      // Find best splits for leaves
      val best_splits = findBestSplits(agg_leaves,
        n_bins,
        n_feature,
        n_label,
        leaves,
        bins,
        impurity,
        min_node_size)

      best_splits.foreach {
        case (pos, best_split) =>
          leaves(pos).split(best_split, max_depth, min_info_gain, min_node_size)
          GBDT.inQueue(node_que, leaves(pos).left_child)
          GBDT.inQueue(node_que, leaves(pos).right_child)
      }
    }

    cart_ps.unpersist()

    root
  }

  /**
    * Method to find best splits for splitting nodes.
    *
    * @param agg_leaves impurity aggregator for splitting nodes
    * @param n_bins     number of bins for all features
    * @param n_feature  number of sub features
    * @param n_label    number of different labels
    * @param leaves     an array stored leaves
    * @param bins       bins of all features
    * @param impurity   an instance of [[Impurity]]
    * @return           (position-id, [[FeatureSplit]])
    */
  def findBestSplits(agg_leaves: RDD[(Int, ImpurityAggregator)],
                     n_bins: Array[Int],
                     n_feature: Int,
                     n_label: Int,
                     leaves: Array[TreeNode],
                     bins: Array[Array[FeatureBin]],
                     impurity: Impurity,
                     min_node_size: Int): Map[Int, NodeBestSplit] = {
    agg_leaves.map {
      case (pos, agg) =>
        agg.toPrefixSum

        val best_split = Range(0, n_feature).flatMap {
          id_f =>
            val n_split = n_bins(id_f) - 1
            Range(0, n_split).map {
              id_s =>
                val (l_impurity, l_predict, l_count) = agg.calLeftInfo(id_f, id_s, n_label)
                val (r_impurity, r_predict, r_count) = agg.calRightInfo(id_f, id_s, n_label)

                val weighted_impurity = impurity.calculate_weighted(l_count, r_count, l_impurity, r_impurity, min_node_size)

                new NodeBestSplit(weighted_impurity,
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
    * @return         an array of [[TreeNode]] which will split next time
    */
  def findCARTNodesToSplit(node_que: mutable.Queue[TreeNode]): Array[TreeNode] = {
    val nodes_builder = mutable.ArrayBuilder.make[TreeNode]
    while (node_que.nonEmpty) {
      nodes_builder += node_que.dequeue()
    }
    nodes_builder.result()
  }

  /**
    * Method to find splits and bins for features.
    *
    * @param train_data training data set represented as a RDD of [[LabeledPoint]]
    * @param n_train    size of training data set
    * @param n_fs       number of features
    * @param n_bins     number of bins for features
    * @return           (Array(Array([[FeatureSplit]]), Array(Array([[FeatureBin]])))
    */
  def findSplitsBins(train_data: RDD[LabeledPoint],
                     n_train: Int,
                     n_fs: Int,
                     n_bins: Array[Int]): (Array[Array[FeatureSplit]], Array[Array[FeatureBin]]) = {

    // Sample the input data to generate splits and bins
    val n_samples = math.max(max_bins * max_bins, bin_samples)
    val r_samples = math.min(n_samples, n_train).toDouble / n_train.toDouble
    val sampled_data = train_data.sample(withReplacement = false, fraction = r_samples).collect()

    val splits = new Array[Array[FeatureSplit]](n_fs)
    val bins = new Array[Array[FeatureBin]](n_fs)

    Range(0, n_fs).foreach {
      id_f =>
        val sampled_f = sampled_data.map(_.fs(id_f))
        val split_vs = findSplitVS(sampled_f, id_f, n_bins)

        val n_split = split_vs.length
        val n_bin = n_split + 1
        n_bins(id_f) = n_bin

        splits(id_f) = new Array[FeatureSplit](n_split)
        bins(id_f) = new Array[FeatureBin](n_bin)

        // Generate splits
        Range(0, n_split).foreach {
          id_split =>
            splits(id_f)(id_split) = new FeatureSplit(id_f, split_vs(id_split))
        }
        // Generate bins
        if (n_bin == 1) {
          bins(id_f)(0) = new FeatureBin(FeatureSplit.lowest(id_f), FeatureSplit.highest(id_f))
        } else {
          bins(id_f)(0) = new FeatureBin(FeatureSplit.lowest(id_f), splits(id_f).head)
          Range(1, n_split).foreach {
            id_bin =>
              bins(id_f)(id_bin) = new FeatureBin(splits(id_f)(id_bin - 1), splits(id_f)(id_bin))
          }
          bins(id_f)(n_split) = new FeatureBin(splits(id_f).last, FeatureSplit.highest(id_f))
        }
    }

    (splits, bins)
  }

  /**
    * Method to find splits for specified feature with sampled training data set.
    *
    * @param sampled_f  sampled value of specified feature
    * @param id_f       ID of specified feature, indexed from 0
    * @param n_bins     number of bins for all features
    * @return           an array stored values of splits for specified feature
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