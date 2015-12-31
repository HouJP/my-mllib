package bda.local.model.tree

import bda.common.obj.LabeledPoint
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{io, Msg, Timer}
import bda.common.Logging
import bda.local.model.tree.Impurity._
import bda.local.model.tree.Loss._
import bda.local.evaluate.Regression.RMSE
import scala.collection.mutable
import bda.common.util.Sampler
import org.apache.commons.math3.distribution.PoissonDistribution

/**
  * External interface of decision tree in standalone.
  */
object DecisionTree {

  /**
    * An adapter for training a decision tree model.
    *
    * @param train_data Training data points.
    * @param valid_data Validation data points.
    * @param impurity Impurity type with String, default is "Variance".
    * @param loss Loss function type with String, default is "SquaredError".
    * @param max_depth Maximum depth of the decision tree, default is 10.
    * @param max_bins Maximum number of bins, default is 32.
    * @param bin_samples Minimum number of samples used in finding splits and bins, default is 10000.
    * @param min_node_size Minimum number of instances in the leaf, default is 15.
    * @param min_info_gain Minimum information gain while splitting, default is 1e-6.
    * @param row_rate sample ratio of train data set.
    * @param col_rate sample ratio of features.
    * @return a [[bda.local.model.tree.DecisionTreeModel]] instance.
    */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint] = null,
            feature_num: Int,
            impurity: String = "Variance",
            loss: String = "SquaredError",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            row_rate: Double = 0.6,
            col_rate: Double = 0.6): DecisionTreeModel = {

    new DecisionTreeTrainer(feature_num,
      Impurity.fromString(impurity),
      Loss.fromString(loss),
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate).train(train_data, valid_data)
  }
}

/**
  * A class which implement decision tree algorithm.
  *
  * @param impurity Impurity type with [[bda.local.model.tree.Impurity]].
  * @param loss Loss function type with [[bda.local.model.tree.Loss]].
  * @param max_depth Maximum depth of the decision tree.
  * @param max_bins Maximum number of bins.
  * @param bin_samples Minimum number of samples used in finding splits and bins.
  * @param min_node_size Minimum number of instances in the leaf.
  * @param min_info_gain Minimum information gain while spliting.
  * @param row_rate sample ratio of train data set.
  * @param col_rate sample ratio of features.
  */
private[tree] class DecisionTreeTrainer(feature_num: Int,
                                        impurity: Impurity,
                                        loss: Loss,
                                        max_depth: Int,
                                        max_bins: Int,
                                        bin_samples: Int,
                                        min_node_size: Int,
                                        min_info_gain: Double,
                                        row_rate: Double,
                                        col_rate: Double) extends Logging {

  /** Impurity calculator */
  val impurity_calculator = impurity match {
    case Variance => VarianceCalculator
    case _ => throw new IllegalAccessException(s"Did not recognize impurity type: $impurity")
  }

  /** Loss calculator */
  val loss_calculator = loss match {
    case SquaredError => SquaredErrorCalculator
    case _ => throw new IllegalArgumentException(s"Did not recognize loss type: $loss")
  }

  /**
   * Method to train a decision tree model over training data.
   *
   * @param train_data Training data which represented as a Sequence of [[bda.common.obj.LabeledPoint]].
   * @param valid_data Validation data which represented as a Sequence of [[bda.common.obj.LabeledPoint]] and can be none.
   * @return A [[bda.local.model.tree.DecisionTreeModel]] instance.
   */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint]): DecisionTreeModel = {

    val timer = new Timer()

    val num_examples = train_data.length
    val num_fs = feature_num
    val num_sub_fs = (num_fs * col_rate).ceil.toInt
    val new_bins = math.min(max_bins, num_examples)
    val num_bins_all = Array.fill[Int](num_fs)(new_bins)

    // find splits and bins for each feature
    val (splits, bins) = findSplitsBins(train_data,
      new_bins,
      num_bins_all,
      num_fs,
      num_examples,
      bin_samples)

    // convert LabeledPoint to TreePoint which used bin-index instead of feature-value
    val dt_points: Seq[DecisionTreePoint] =
      DecisionTreePoint.convertToDecisionTreeRDD(train_data, bins, num_bins_all, num_fs, row_rate)
    //Log.log("INFO", "Training data set convert to DTreePoint done.")

    // check sampling ratio of training data set
    // val ck_row_rate = dt_points.map(_.weight).sum.toDouble / num_examples
    // println(s"HouJP >> sampling ratio of training data set = $ck_row_rate")

    // create root for decision tree
    val root = DecisionTreeNode.empty(id = 1, 0)
    // calculate root's impurity and root's predict
    //    val root_count = num_examples
    //    val root_sum = dt_points.map(_.label).sum
    //    val root_squared_sum = dt_points.map(p => p.label * p.label).sum
    val root_count = dt_points.map(_.weight).sum
    val root_sum = dt_points.map(p => p.label * p.weight).sum
    val root_squared_sum = dt_points.map(p => p.label * p.label * p.weight).sum

    root.count = root_count
    root.impurity = impurity_calculator.calculate(root_count, root_sum, root_squared_sum)
    root.predict = loss_calculator.predict(root_sum, root_count)
    root.sampleFeatures(num_fs, col_rate)

    // create a node queue which help to generate a binary tree
    val node_que = new mutable.Queue[DecisionTreeNode]()
    node_que.enqueue(root)

    while (node_que.nonEmpty) {
      val leaves = findSplittingNodes(node_que)

      val num_leaves = leaves.length
      val id_leaves = leaves.map { case node =>
        node.id
      }
      val ind_leaves = id_leaves.zipWithIndex.toMap

      //val agg_leaves = dt_points.mapPartitions { points =>
      val agg_leaves = {
        val agg_leaves = Array.tabulate(num_leaves)(index => new DecisionTreeStatsAgg(num_bins_all, num_fs, num_sub_fs))

        dt_points.foreach { p =>
          val id_leaf = DecisionTreeTrainer.findLeafId(p, root, bins)
          if (ind_leaves.contains(id_leaf)) {
            val ind_leaf = ind_leaves(id_leaf)
            agg_leaves(ind_leaf).update(p, leaves(ind_leaf).sub_fs)
          }
        }

        agg_leaves.view.zipWithIndex.map(_.swap)
      }

      //  agg_leaves.view.zipWithIndex.map(_.swap).iterator
      //}.reduceByKey((a, b) => a.merge(b))

      val best_splits = findBestSplit(agg_leaves,
        leaves,
        num_fs,
        num_sub_fs,
        num_bins_all,
        impurity_calculator,
        loss_calculator)

      updateBestSplit(leaves,
        best_splits,
        bins,
        max_depth,
        min_info_gain,
        min_node_size,
        num_fs,
        col_rate)

      leaves.foreach { node =>
        inQueue(node_que, node.left_child)
        inQueue(node_que, node.right_child)
      }
    }

    // calculate rmse
    val train_rmse = evaluate(root, train_data)
    val valid_rmse = if (null != valid_data) {
      evaluate(root, valid_data)
    } else {
      null
    }

    // get time cost
    val time_cost = timer.cost()

    // show logs
    val msg = Msg("RMSE(train)" -> train_rmse)
    if (null != valid_data) {
      msg.append("RMSE(valid)", valid_rmse)
    }
    msg.append("time cost", time_cost)
    logInfo(msg.toString)

    new DecisionTreeModel(root,
      feature_num,
      impurity,
      loss,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      impurity_calculator,
      loss_calculator)
  }

  /**
   * Method to get all the nodes in the queue.
   *
   * @param node_que The queue stored all splitting nodes.
   * @return Array of [[bda.local.model.tree.DecisionTreeNode]].
   */
  def findSplittingNodes(node_que: mutable.Queue[DecisionTreeNode]): Array[DecisionTreeNode] = {
    val leaves_builder = mutable.ArrayBuilder.make[DecisionTreeNode]
    while (node_que.nonEmpty) {
      // println(s"HouJP >> sub fs of ${node_que.head.id} is: ${node_que.head.sub_fs.getOrElse(Seq[Int]()).mkString(",")}")
      leaves_builder += node_que.dequeue()
    }
    leaves_builder.result()
  }


  /**
   * Method to find best splits and bins for each features.
   *
   * @param input Training data which represented as a RDD [[bda.common.obj.LabeledPoint]].
   * @param new_bins Maximum possible number of bins.
   * @param num_bins_all Number of bins for each features.
   * @param num_features Number of features.
   * @param num_examples Size of training data.
   * @param min_samples Minimum possible number of samples.
   * @return A tuple of [[bda.local.model.tree.DecisionTreeSplit]] and [[bda.local.model.tree.DecisionTreeBin]].
   */
  def findSplitsBins(input: Seq[LabeledPoint],
                     new_bins: Int,
                     num_bins_all: Array[Int],
                     num_features: Int,
                     num_examples: Int,
                     min_samples: Int): (Array[Array[DecisionTreeSplit]], Array[Array[DecisionTreeBin]]) = {

    // sample the input
    val required_samples = math.max(new_bins * new_bins, min_samples)
    val fraction = if (required_samples < num_examples) {
      required_samples.toDouble / num_examples.toDouble
    } else {
      1.0
    }
    val sampled_input = Sampler.withoutBack[LabeledPoint](input, fraction)//input.sample(withReplacement = false, fraction, new Random().nextLong()).collect()
    // println("HouJP >> sampled input:")
    // sampled_input.foreach(println)

    val splits = new Array[Array[DecisionTreeSplit]](num_features)
    val bins = new Array[Array[DecisionTreeBin]](num_features)

    var index_f = 0
    while (index_f < num_features) {
      val sampled_fs = sampled_input.map(lp => lp.fs(index_f))
      val feature_splits = findSplits(sampled_fs, num_bins_all, index_f)

      val num_splits = feature_splits.length
      val num_bins = num_splits + 1

      splits(index_f) = new Array[DecisionTreeSplit](num_splits)
      bins(index_f) = new Array[DecisionTreeBin](num_bins)

      var index_split = 0
      while (index_split < feature_splits.length) {
        splits(index_f)(index_split) = new DecisionTreeSplit(index_f, feature_splits(index_split))
        index_split += 1
      }

      bins(index_f)(0) = new DecisionTreeBin(new DecisionTreeLowestSplit(index_f), splits(index_f).head)
      var index_bin = 1
      while (index_bin < feature_splits.length) {
        bins(index_f)(index_bin) = new DecisionTreeBin(splits(index_f)(index_bin - 1), splits(index_f)(index_bin))
        index_bin += 1
      }
      bins(index_f)(feature_splits.length) = new DecisionTreeBin(splits(index_f).last, new DecisionTreeHighestSplit(index_f))

      index_f += 1
    }

    (splits, bins)
  }


  /**
   * Method to find splits for the feature.
   *
   * @param sampled_features Values of the feature.
   * @param num_bins_all Number of bins for each features.
   * @param index_f Id of the feature.
   * @return value of splits for the feature with id = index_f.
   */
  def findSplits(sampled_features: Seq[Double],
                 num_bins_all: Array[Int],
                 index_f: Int): Seq[Double] = {

    val splits = {
      val num_splits = DecisionTreeTrainer.numSplits(index_f, num_bins_all)

      // get count for each distinct value
      val values = sampled_features.foldLeft(Map.empty[Double, Int]) { (m, x) =>
        m + ((x, m.getOrElse(x, 0) + 1))
      }.toSeq.sortBy(_._1).toArray

      // if possible splits is not enough or just enough, just return all possible splits
      val possible_splits = values.length
      //Log.log("INFO", s"<findSplit> possible_splits = $possible_splits")
      if (possible_splits <= num_splits) {
        values.map(_._1).take(possible_splits - 1)
      } else {
        val stride: Double = sampled_features.length.toDouble / (num_splits + 1)

        val split_builder = mutable.ArrayBuilder.make[Double]
        var index = 1
        var cur_cnt = values(0)._2
        var target_cnt = stride
        while (index < values.length) {
          val pre_cnt = cur_cnt
          cur_cnt += values(index)._2
          val pre_gap = math.abs(pre_cnt - target_cnt)
          val cur_gap = math.abs(cur_cnt - target_cnt)
          if (pre_gap < cur_gap) {
            split_builder += values(index - 1)._1
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
    require(splits.length > 0, s"DTree could not handle feature $index_f since it had only 1 unique value." +
      " Please remove this feature and try again.")

    // reset features' bin-number which maybe changed
    DecisionTreeTrainer.setBins(index_f, splits.length, num_bins_all)

    splits
  }

  /**
   * Method to find best splits for splitting nodes.
   *
   * @param agg_leaves Aggregators for splitting nodes represented as a RDD of [[bda.local.model.tree.DecisionTreeStatsAgg]]
   * @param num_fs Number of features.
   * @param num_bins_all Number of bins for all features.
   * @param impurity_calculator Impurity calculator.
   * @param loss_calculator Loss calculator.
   * @return Map[node-id, best-split].
   */
  def findBestSplit(agg_leaves: Seq[(Int, DecisionTreeStatsAgg)],
                    leaves: Seq[DecisionTreeNode],
                    num_fs: Int,
                    num_sub_fs: Int,
                    num_bins_all: Array[Int],
                    impurity_calculator: ImpurityCalculator,
                    loss_calculator: LossCalculator): scala.collection.Map[Int, DecisionTreeBestSplit] = {

    agg_leaves.map { case (ind, agg) =>
      agg.toPrefixSum()

      val best_split = Range(0, num_sub_fs).map { ind_sub_f =>
        val index_f = leaves(ind).sub_fs(ind_sub_f)
        val num_splits = DecisionTreeTrainer.numSplits(index_f, num_bins_all)
        Range(0, num_splits).map { index_b =>
          val (l_impurity, l_pred, l_cnt) = agg.calLeftInfo(index_f, index_b, impurity_calculator, loss_calculator)
          val (r_impurity, r_pred, r_cnt) = agg.calRightInfo(index_f, index_b, impurity_calculator, loss_calculator)
          val f_cnt = l_cnt + r_cnt

          val weighted_impurity = l_impurity * l_cnt / f_cnt + r_impurity * r_cnt / f_cnt


          new DecisionTreeBestSplit(weighted_impurity,
            index_f,
            index_b,
            l_impurity,
            r_impurity,
            l_pred,
            r_pred,
            l_cnt,
            r_cnt)
        }.minBy(_.weighted_impurity)
      }.minBy(_.weighted_impurity)
      (ind, best_split)
    }.toMap
  }

  /**
   * Grow the decision tree according to best splits of each node.
   *
   * @param leaves Array stored splitting nodes.
   * @param best_splits Best splits of all splitting nodes.
   * @param bins Bins of all features.
   * @param max_depth Maximum depth of the decision tree.
   * @param min_info_gain Minimum information gain while splitting.
   * @param min_node_size Minimum size of node.
   */
  def updateBestSplit(leaves: Array[DecisionTreeNode],
                      best_splits: scala.collection.Map[Int, DecisionTreeBestSplit],
                      bins: Array[Array[DecisionTreeBin]],
                      max_depth: Int,
                      min_info_gain: Double,
                      min_node_size: Int,
                      num_fs: Int,
                      col_rate: Double): Unit = {

    val num_leaves = leaves.length

    var index = 0
    while (index < num_leaves) {
      val node = leaves(index)
      val best_split = best_splits(index)
      val info_gain = node.impurity - best_split.weighted_impurity
      val split = bins(best_split.index_f)(best_split.index_b).high_split

      // println(s"HouJP >> $node, l_cnt = ${best_split.l_cnt}, r_cnt = ${best_split.r_cnt}")

      if (((node.impurity - best_split.weighted_impurity) >= min_info_gain)
        && (node.depth < max_depth)
        && (node.count > min_node_size)) { // Is it reasonable of judging min_node_size?

        // println("HouJP >> splitting ...")

        node.is_leaf = false
        node.split = Some(split)
        node.generate_lchild(best_split.l_impurity, best_split.l_pred, best_split.l_cnt, num_fs, col_rate)
        node.generate_rchild(best_split.r_impurity, best_split.r_pred, best_split.r_cnt, num_fs, col_rate)
      }

      index += 1
    }
  }

  /**
   * If the splitting node has children, then push into the queue as a new splitting node.
   *
   * @param que Queue which stored splitting node.
   * @param node Left or right child of the splitting node.
   */
  def inQueue(que: mutable.Queue[DecisionTreeNode], node: Option[DecisionTreeNode]): Unit = {
    node match {
      case Some(n)  => que.enqueue(n)
      case _ =>
    }
  }

  /**
   * Calculate the RMSE for the input data.
   * @param root Root node of the decision tree structure.
   * @param input Input data represented as RDD of [[bda.common.obj.LabeledPoint]].
   * @return RMSE of the input data.
   */
  def evaluate(root: DecisionTreeNode, input: Seq[LabeledPoint]): Double = {
    val lps = input.map { case lp =>
      val pred = DecisionTreeModel.predict(lp.fs, root)
      (lp.label, pred)
    }

    RMSE(lps)
  }
}


/**
 * Decision tree model for regression.
 *
 * @param root Root node of decision tree structure.
 * @param impurity Impurity type with [[bda.local.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.local.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param max_bins Maximum number of bins.
 * @param min_samples Minimum number of samples used in finding splits and bins.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param row_rate sampling rate of training data set.
 * @param col_rate sampling rate of features.
 * @param impurity_calculator Impurity calculator.
 * @param loss_calculator Loss calculator.
 */
@SerialVersionUID(6529125048396757390L)
class DecisionTreeModel(val root: DecisionTreeNode,
                        val feature_num: Int,
                        val impurity: Impurity,
                        val loss: Loss,
                        val max_depth: Int,
                        val max_bins: Int,
                        val min_samples: Int,
                        val min_node_size: Int,
                        val min_info_gain: Double,
                        val row_rate: Double,
                        val col_rate: Double,
                        val impurity_calculator: ImpurityCalculator,
                        val loss_calculator: LossCalculator) extends Serializable {

  /**
   * Predict values for the given data using the model trained.
   *
   * @param input Prediction data set which represented as a RDD of [[bda.common.obj.LabeledPoint]].
   * @return A RDD stored prediction.
   */
  def predict(input: Seq[LabeledPoint]): Seq[Double] = {
    val root = this.root
    val loss = this.loss_calculator

    input.map { case lp =>
      val pred = DecisionTreeModel.predict(lp.fs, root)
      pred
    }

    //Log.log("INFO", s"predict done, with RMSE = ${math.sqrt(err)}")
  }

  /**
    * Store decision tree model on the disk.
    *
    * @param path path of the location on the disk
    */
  def save(path: String): Unit = {
    DecisionTreeModel.save(path, this)
  }

  def showStructure: Unit = {
    val node_que = new mutable.Queue[DecisionTreeNode]()
    node_que.enqueue(root)
    while (node_que.nonEmpty) {

      if (!node_que.head.is_leaf) {
        node_que.enqueue(node_que.head.left_child.get)
        node_que.enqueue(node_que.head.right_child.get)
      }

      node_que.dequeue()
    }
  }
}

/**
 * Class to store best splitting information.
 *
 * @param weighted_impurity Weighted impurity of left child and right child.
 * @param index_f Id of the feature.
 * @param index_b Id of the bin.
 * @param l_impurity Impurity of left child.
 * @param r_impurity Impurity of right child.
 * @param l_pred Prediction of left child.
 * @param r_pred Prediction of right child.
 * @param l_cnt Number of instances of left child.
 * @param r_cnt Number of instances of right child.
 */
private[tree] case class DecisionTreeBestSplit(weighted_impurity: Double,
                                               index_f: Int,
                                               index_b: Int,
                                               l_impurity: Double,
                                               r_impurity: Double,
                                               l_pred: Double,
                                               r_pred: Double,
                                               l_cnt: Int,
                                               r_cnt: Int) extends Serializable

object DecisionTreeModel {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point.
   * @param root root node of decision tree structure.
   * @return predicted value from the trained model.
   */
  private[tree] def predict(fs: SparseVector[Double], root: DecisionTreeNode): Double = {
    var node = root
    while (!node.is_leaf) {
      if (fs(node.split.get.feature) <= node.split.get.threshold) {
        node = node.left_child.get
      } else {
        node = node.right_child.get
      }
    }
    node.predict
  }

  /**
    * Store decision tree model on the disk.
    *
    * @param pt path of the location on the disk
    * @param model decision tree model
    */
  def save(pt: String, model: DecisionTreeModel): Unit = {

    io.writeObject[DecisionTreeModel](pt, model)
  }

  /**
    * Load decision tree model from the disk.
    *
    * @param pt path of the localtion on the disk
    */
  def load(pt: String): DecisionTreeModel = {

    io.readObject[DecisionTreeModel](pt)
  }
}


/**
 * Static functions of [[bda.local.model.tree.DecisionTreeTrainer]].
 */
private[tree] object DecisionTreeTrainer {

  /**
   * Method to find the splitting node id which contains the point.
   * @param point A instance in training data set.
   * @param root Root of the decision tree.
   * @param bins Array of bins of all the features.
   * @return Id of splitting node which contains the point.
   */
  def findLeafId(point: DecisionTreePoint, root: DecisionTreeNode, bins: Array[Array[DecisionTreeBin]]): Int = {
    var leaf = root
    while (!leaf.is_leaf) {
      val split = leaf.split.get
      if (bins(split.feature)(point.binned_fs(split.feature)).high_split.threshold <= split.threshold) {
        leaf = leaf.left_child.get
      } else {
        leaf = leaf.right_child.get
      }
    }
    leaf.id
  }

  /**
   * Method to get number of splits with specified feature id.
   *
   * @param index_feature Id of feature.
   * @param num_bins Number of bins of all features.
   * @return Number of splits with feature-id = index_feature.
   */
  def numSplits(index_feature: Int, num_bins: Seq[Int]): Int = {
    num_bins(index_feature) - 1
  }

  /**
   * Method to set number of bins with specified feature id.
   * @param index_feature Id of feature.
   * @param num_splits Number of bins of all features.
   * @param num_bins Number of bins of all features.
   */
  def setBins(index_feature: Int, num_splits: Int, num_bins: Array[Int]): Unit = {
    num_bins(index_feature) = num_splits + 1
  }
}


/**
 * Class that represents the features and labels of a data point.
 *
 * @param label Label of the data point.
 * @param binned_fs features of the data point.
 */
private[tree] case class DecisionTreePoint(label: Double, weight: Int, binned_fs: Array[Int]) extends Serializable

private[tree] object DecisionTreePoint {

  /**
   * Convert input to a RDD of [[bda.local.model.tree.DecisionTreePoint]].
   * @param input A RDD of [[bda.common.obj.LabeledPoint]].
   * @param bins Bins of all features.
   * @param num_bins_all Number of bins of all features.
   * @param num_fs Number of features.
   * @param row_rate sample rate of training data set.
   * @return A RDD of [[bda.local.model.tree.DecisionTreePoint]].
   */
  def convertToDecisionTreeRDD(input: Seq[LabeledPoint],
                               bins: Array[Array[DecisionTreeBin]],
                               num_bins_all: Array[Int],
                               num_fs: Int,
                               row_rate: Double): Seq[DecisionTreePoint] = {

    // set input instances' weights use poisson distribution
    val poisson = new PoissonDistribution(row_rate)

    input.map{ case lp =>
      DecisionTreePoint.convertToDecisionTreePoint(lp, bins, num_bins_all, num_fs, poisson)
    }
  }

  /**
   * Convert a data point into [[bda.local.model.tree.DecisionTreePoint]].
   * @param lp A [[bda.common.obj.LabeledPoint]] instance.
   * @param bins Bins of all features.
   * @param num_bins_all Number of bins of all features.
   * @param num_fs Number of fetures.
   * @param poisson poisson distribution to sample without replacement.
   * @return A [[bda.local.model.tree.DecisionTreePoint]] instance.
   */
  def convertToDecisionTreePoint(lp: LabeledPoint,
                                 bins: Array[Array[DecisionTreeBin]],
                                 num_bins_all: Array[Int],
                                 num_fs: Int,
                                 poisson: PoissonDistribution): DecisionTreePoint = {

    val binned_fs = new Array[Int](num_fs)
    for (index_f <- 0 until num_fs) {
      val binned_f = binarySearchForBin(lp.fs(index_f), index_f, bins, num_bins_all, num_fs)

      // check the binned feature
      require(-1 != binned_f, s"Point with label = ${lp.label} couldn't find correct bin" +
        s" with feature-index = $index_f, feature-value = ${lp.fs(index_f)}")

      // convert the feature to binned feature
      binned_fs(index_f) = binned_f
    }

    // get weight use poisson distribution
    val weight = {
      if (poisson.getMean < 1.0) {
        poisson.sample()
      } else {
        1
      }
    }

    new DecisionTreePoint(lp.label, weight, binned_fs)
  }

  /**
   * Method to find bin-id by binary search with specified feature-value and feature-id.
   * @param value Value of specified feature.
   * @param index_feature Id of feature.
   * @param bins Bins of all features.
   * @param num_bins_all Number of bins of all features.
   * @param num_fs Number of features.
   * @return Id of bin with specified feature.
   */
  def binarySearchForBin(value: Double,
                         index_feature: Int,
                         bins: Array[Array[DecisionTreeBin]],
                         num_bins_all: Array[Int],
                         num_fs: Int): Int = {

    var index_l = 0
    var index_r = num_bins_all(index_feature) - 1
    while (index_l <= index_r) {
      val index_m = (index_l + index_r) / 2
      if (bins(index_feature)(index_m).low_split.threshold >= value) {
        index_r = index_m - 1
      } else if (bins(index_feature)(index_m).high_split.threshold < value) {
        index_l = index_m + 1
      } else {
        return index_m
      }
    }
    -1
  }
}


/**
 * Class of nodes which form a tree.
 *
 * @param id Integer node id, 1-based.
 * @param depth Node depth in a tree, 0-based.
 */
private[tree] class DecisionTreeNode (val id: Int,
                                      val depth: Int) extends Serializable {

  /** flag to show whether is a leaf-node */
  var is_leaf: Boolean = true
  /** impurity value of this node */
  var impurity: Double = 0.0
  /** prediction value of this node */
  var predict: Double = 0.0
  /** splitting way of this node */
  var split: Option[DecisionTreeSplit] = None
  /** left child */
  var left_child: Option[DecisionTreeNode] = None
  /** right child */
  var right_child: Option[DecisionTreeNode] = None
  /** number of instances */
  var count: Int = 0
  /** sub features */
  var sub_fs: Seq[Int] = null

  /**
   * Convert this node into a string.
   *
   * @return A string which represented this node.
   */
  override def toString: String = {
    s"Node: id = $id, depth = $depth, predict = $predict, count = $count, impurity = $impurity, split = {${split.getOrElse("None")}}"
  }

  /**
   * sample features.
   *
   * @param tol total number of features.
   * @param rate sampling ratio of features.
   */
  def sampleFeatures(tol: Int, rate: Double): Unit = {
    sub_fs = Sampler.withoutBack(tol, rate)
    // println(s"HouJP >> <sampleFeatures> tol = $tol, rate = $rate, sub_num = ${sub_fs.getOrElse(Seq[Int]()).length}")
  }

  /**
   * Method to generate left child of this node.
   * @param l_impurity impurity value of left child of this node.
   * @param l_pred prediction value of left child of this node.
   * @param num_fs number of features.
   * @param col_rate sampling rate of features.
   */
  def generate_lchild(l_impurity: Double, l_pred: Double, l_cnt: Int, num_fs: Int, col_rate: Double): Unit = {
    left_child = Some(DecisionTreeNode.empty(id << 1, depth + 1))
    left_child.get.count = l_cnt
    left_child.get.impurity = l_impurity
    left_child.get.predict = l_pred
    left_child.get.sampleFeatures(num_fs, col_rate)
  }

  /**
   * Method to generate right child of this node.
   * @param r_impurity impurity value of right child of this node.
   * @param r_pred prediction value of right child of this node.
   * @param num_fs number of features.
   * @param col_rate sampling rate of features.
   */
  def generate_rchild(r_impurity: Double, r_pred: Double, r_cnt: Int, num_fs: Int, col_rate: Double): Unit = {
    right_child = Some(DecisionTreeNode.empty(id << 1 | 1, depth + 1))
    right_child.get.count = r_cnt
    right_child.get.impurity = r_impurity
    right_child.get.predict = r_pred
    right_child.get.sampleFeatures(num_fs, col_rate)
  }

}

private[tree] object DecisionTreeNode {

  /**
   * Method to generate an empty node with specified id and depth.
   *
   * @param id Id of new node.
   * @param depth Depth of new node.
   * @return A [[bda.local.model.tree.DecisionTreeNode]] instance.
   */
  def empty(id: Int, depth: Int): DecisionTreeNode = {
    new DecisionTreeNode(id, depth)
  }
}

/**
 * Class of aggregator to statistic node number, summation and squared summation.
 *
 * @param num_bins_all Number of bins of all features.
 * @param num_fs Number of features.
 */
private[tree] class DecisionTreeStatsAgg(num_bins_all: Array[Int],
                                         num_fs: Int,
                                         num_sub_fs: Int) extends Serializable {

  /** Step size of the state */
  val step_stat = 3
  /** Number of states */
  val num_stats = num_bins_all.sum
  /** Length of array which stored states */
  val len_stats = step_stat * num_stats
  /** Array which stored states */
  val stats = new Array[Double](len_stats)
  /** offset of each features */
  val off_fs = {
    num_bins_all.scanLeft(0)((sum, num_bins) => sum + num_bins * step_stat)
  }

  /**
   * Method to update states array with specified feature.
   *
   * @param label Label which will add into the states.
   * @param index_f Id of feature.
   * @param binned_f Binned feature value.
   */
  def update(label: Double, weight: Int, index_f: Int, binned_f: Int): Unit = {
    val offset = off_fs(index_f) + 3 * binned_f
    stats(offset + 0) += weight
    stats(offset + 1) += label * weight
    stats(offset + 2) += label * label * weight
  }

  /**
   * Method to update sates array with a data point.
   *
   * @param p A data point used to udpate states.
   * @param sub_fs Sub features of specified node.
   */
  def update(p: DecisionTreePoint, sub_fs: Seq[Int]): Unit = {
    var index = 0
    while (index < num_sub_fs) {
      val index_f = sub_fs(index)
      update(p.label, p.weight, index_f, p.binned_fs(index_f))
      index += 1
    }
  }

  /**
   * Methed to merge another aggregator into this aggregator.
   *
   * @param other another aggregator.
   * @return this aggregator which after merging.
   */
  def merge(other: DecisionTreeStatsAgg): DecisionTreeStatsAgg = {
    var index = 0
    while (index < len_stats) {
      stats(index) += other.stats(index)
      index += 1
    }

    this
  }

  /**
   * Method to convert value of states in the array into prefix summation form.
   *
   * @return this aggregator which represented as prefix summation.
   */
  def toPrefixSum(): DecisionTreeStatsAgg = {
    Range(0, num_fs).foreach { index_f =>
      val num_bin = num_bins_all(index_f)

      var index_b = 1
      while (index_b < num_bin) {
        val offset = off_fs(index_f) + 3 * index_b
        stats(offset + 0) += stats(offset + 0 - 3)
        stats(offset + 1) += stats(offset + 1 - 3)
        stats(offset + 2) += stats(offset + 2 - 3)
        index_b += 1
      }
    }
    this
  }

  /**
   * Method to calculate left child impurity, prediction and count.
   *
   * @param index_f Id of the feature.
   * @param index_b Id of the bins.
   * @param impurity_calculator Impurity calculator.
   * @param loss_calculator Loss calculator.
   * @return (impurity, predict, count) of left child.
   */
  def calLeftInfo(index_f: Int,
                  index_b: Int,
                  impurity_calculator: ImpurityCalculator,
                  loss_calculator: LossCalculator): (Double, Double, Int) = {

    val off = off_fs(index_f) + 3 * index_b

    val count = stats(off).toInt
    val sum = stats(off + 1)
    val squared_sum = stats(off + 2)

    val impurity = impurity_calculator.calculate(count, sum, squared_sum)
    val predict = loss_calculator.predict(sum, count)

    (impurity, predict, count)
  }

  /**
   * Method to calculate right child impurity, prediction and count.
   *
   * @param index_f Id of the feature.
   * @param index_b Id of the bins.
   * @param impurity_calculator Impurity calculator.
   * @param loss_calculator Loss calculator.
   * @return (impurity, predict, count) of right child.
   */
  def calRightInfo(index_f: Int,
                   index_b: Int,
                   impurity_calculator: ImpurityCalculator,
                   loss_calculator: LossCalculator): (Double, Double, Int) = {

    val off = off_fs(index_f) + 3 * index_b
    val last_off = off_fs(index_f) + 3 * (num_bins_all(index_f) - 1)

    val count = (stats(last_off) - stats(off)).toInt
    val sum = stats(last_off + 1) - stats(off + 1)
    val squared_sum = stats(last_off + 2) - stats(off + 2)

    val impurity = impurity_calculator.calculate(count, sum, squared_sum)
    val predict = loss_calculator.predict(sum, count)

    (impurity, predict, count)
  }
}

/**
 * Class of Split which stored information of splitting.
 *
 * @param feature Id of feature.
 * @param threshold Threshold used in splitting. Split left if feature < threshold, else right.
 */
private[tree] case class DecisionTreeSplit(feature: Int, threshold: Double) {

  /**
   * Method to convert this into a string.
   *
   * @return A string which represented this split.
   */
  override def toString: String = {
    s"feature = $feature, threshold = $threshold"
  }
}

/**
 * Class of Split which is located at left most.
 *
 * @param feature Id of feature.
 */
private[tree] class DecisionTreeLowestSplit(feature: Int) extends DecisionTreeSplit(feature, Double.MinValue)

/**
 * Class of Split which is located at right most.
 *
 * @param feature Id of feature.
 */
private[tree] class DecisionTreeHighestSplit(feature: Int) extends DecisionTreeSplit(feature, Double.MaxValue)

/**
 * Class of Bin which stored information of borders.
 *
 * @param low_split Left border of this bin.
 * @param high_split Right border of this bin.
 */
private[tree] case class DecisionTreeBin(low_split: DecisionTreeSplit, high_split: DecisionTreeSplit)