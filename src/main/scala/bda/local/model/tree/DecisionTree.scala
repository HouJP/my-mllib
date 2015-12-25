package bda.local.model.tree

import bda.common.obj.LabeledPoint
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{io, Msg, Timer}
import bda.common.Logging
import bda.local.model.tree.Impurity._
import bda.local.model.tree.Loss._
import bda.local.evaluate.Regression.RMSE

import scala.collection.mutable
import scala.util.Random

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
   * @param max_depth Maximum depth of the decision tree, default is 6.
   * @param min_node_size Minimum number of instances in the leaf, default is 15.
   * @param min_info_gain Minimum information gain while spliting, default is 1e-6.
   * @return a [[bda.local.model.tree.DecisionTreeModel]] instance.
   */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint] = null,
            feature_num: Int,
            impurity: String = "Variance",
            loss: String = "SquaredError",
            max_depth: Int = 10,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6): DecisionTreeModel = {

    val model = new DecisionTreeTrainer(feature_num,
      Impurity.fromString(impurity),
      Loss.fromString(loss),
      max_depth,
      min_node_size,
      min_info_gain).train(train_data, valid_data)

    model
  }
}

/**
 * A class which implement decision tree algorithm.
 *
 * @param impurity Impurity type with [[bda.local.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.local.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 */
private[tree] class DecisionTreeTrainer(feature_num: Int,
                                        impurity: Impurity,
                                        loss: Loss,
                                        max_depth: Int,
                                        min_node_size: Int,
                                        min_info_gain: Double) extends Logging {

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
   * @param train_data Training data which represented as an array of [[bda.common.obj.LabeledPoint]].
   * @param valid_data Validation data which represented as an array of [[bda.common.obj.LabeledPoint]] and can be none.
   * @return A [[bda.local.model.tree.DecisionTreeModel]] instance.
   */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint] = null): DecisionTreeModel = {

    val timer = new Timer()

    val num_f = feature_num
    val num_d = train_data.length

    val index_d = new Array[Int](num_d)
    for (ind <- 0 until num_d) {
      index_d(ind) = ind
    } // zip with index

    // stat of the root
    val index_l = 0
    val index_r = num_d // topNode covers data from 0 until numData: [0, numData)
    val count = index_r - index_l
    var sum = 0.0
    var sum_squares = 0.0
    for (index <- index_l until index_r) {
      val value = train_data(index).label
      sum += value
      sum_squares += value * value
    }
    val stat = new DecisionTreeStat(
      impurity_calculator,
      count,
      sum,
      sum_squares,
      index_l,
      index_r) // nodeinfo

    // root of the decision tree
    val pred = loss_calculator.predict(stat)
    val root = DecisionTreeNode.empty(nodeIndex = 1, nodeDep = 0, predict = pred)

    val que_node = new mutable.Queue[(DecisionTreeNode, DecisionTreeStat)]
    que_node.enqueue((root, stat)) // topNode covers dataIndex [0, numData)

    while (que_node.nonEmpty) {
      findBestSplit(train_data, index_d, que_node, num_f)
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
      min_node_size,
      min_info_gain,
      impurity_calculator,
      loss_calculator)
  }

  /**
   * Find the best threshold to split the front node in the queue.
   *
   * @param input training data
   * @param index_d the index array of the training data
   * @param que_node nodes queue which stored the nodes need to be splited
   * @param num_f training data metadata and the strategy of the decision tree
   */
  def findBestSplit(input: Seq[LabeledPoint],
                    index_d: Array[Int],
                    que_node: mutable.Queue[(DecisionTreeNode, DecisionTreeStat)],
                    num_f: Int): Unit = {

    val (node, stat) = que_node.dequeue()

    // check node, split it if necessary
    if (!checkNode(node, stat)) {
      node.isLeaf = true

      //Log.log("INFO", s"node_${node.id} stop split")
      return
    }

    // find the best split
    var best_sp_v = 0.0 // best split value
    var best_f_id = 0 // best feature id
    var max_ig = min_info_gain
    var best_stat_l = DecisionTreeStat.empty
    var best_stat_r = DecisionTreeStat.empty
    for (featureIndex <- 0 until num_f) {
      var a_best_sp_v = 0.0
      var a_max_ig = max_ig
      var a_best_stat_l: DecisionTreeStat = DecisionTreeStat.empty
      var a_best_stat_r: DecisionTreeStat = DecisionTreeStat.empty

      val stat_l = new DecisionTreeStat(
        impurity_calculator,
        0,
        0,
        0,
        stat.left_index,
        stat.left_index)
      val stat_r = new DecisionTreeStat(
        impurity_calculator,
        stat.count,
        stat.sum,
        stat.squared_sum,
        stat.left_index,
        stat.right_index)

      val f_v = new Array[Double](stat.count)
      val index_tmp = new Array[Int](stat.count)
      for (offset <- 0 until stat.count) {
        f_v(offset) = input(index_d(stat.left_index + offset)).fs(featureIndex)
        index_tmp(offset) = index_d(stat.left_index + offset)
      }
      val ordered = index_tmp.zip(f_v).sortBy(_._2)

      for (offset <- 0 until (stat.count - 1)) {
        val dataValue = input(ordered(offset)._1).label
        stat_l :+= dataValue
        dataValue -=: stat_r

        if (ordered(offset + 1)._2 > ordered(offset)._2) {
          val sp_v = (ordered(offset + 1)._2 + ordered(offset)._2) / 2
          val ig = stat.impurity -
            (1.0 * stat_l.count / stat.count) * stat_l.impurity -
            (1.0 * stat_r.count / stat.count) * stat_r.impurity

          if (ig > a_max_ig) {
            a_max_ig = ig
            a_best_sp_v = sp_v
            a_best_stat_l.copy(stat_l)
            a_best_stat_r.copy(stat_r)
          }
        }
      }

      if (a_max_ig > max_ig) {
        max_ig = a_max_ig
        best_sp_v = a_best_sp_v
        best_f_id = featureIndex
        best_stat_l.copy(a_best_stat_l)
        best_stat_r.copy(a_best_stat_r)

        for (dataOffset <- 0 until stat.count) {
          index_d(stat.left_index + dataOffset) = ordered(dataOffset)._1
        }
      }
    }

    // add leftChild and rightChild to the queue
    if (max_ig > min_info_gain) {
      val ind_l = DecisionTreeNode.leftChildIndex(node.id)
      val ind_r = DecisionTreeNode.rightChildIndex(node.id)
      val pre_l = loss_calculator.predict(best_stat_l)
      val pre_r = loss_calculator.predict(best_stat_r)
      val node_l = DecisionTreeNode.empty(ind_l, node.dep + 1, pre_l)
      val node_r = DecisionTreeNode.empty(ind_r, node.dep + 1, pre_r)

      node.splitValue = best_sp_v
      node.featureID = best_f_id
      node.leftNode = Option(node_l)
      node.rightNode = Option(node_r)

      que_node.enqueue((node_l, best_stat_l))
      que_node.enqueue((node_r, best_stat_r))

    } else {
      node.isLeaf = true
    }
  }

  /**
   * Check the node whether or not splits
   * @param node the node waiting to split
   * @param stat the node status
   * @return whether or not splits
   */
  def checkNode(node: DecisionTreeNode, stat: DecisionTreeStat): Boolean = {
    if (node.dep >= max_depth) {
      return false
    }
    if (stat.count <= min_node_size) {
      return false
    }
    return true
  }

  /**
   * Calculate the RMSE for the input data.
   * @param root Root node of the decision tree structure.
   * @param input Input data represented as array of [[bda.common.obj.LabeledPoint]].
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
 * Static functions of [[bda.spark.model.tree.DecisionTreeTrainer]].
 */
private[tree] object DecisionTreeTrainer {


}

/**
 * Decision tree model for regression.
 *
 * @param root Root node of decision tree structure.
 * @param impurity Impurity type with [[bda.spark.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.spark.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param impurity_calculator Impurity calculator.
 * @param loss_calculator Loss calculator.
 */
@SerialVersionUID(6529125048396757390L)
class DecisionTreeModel(val root: DecisionTreeNode,
                        val feature_num: Int,
                        val impurity: Impurity,
                        val loss: Loss,
                        val max_depth: Int,
                        val min_node_size: Int,
                        val min_info_gain: Double,
                        val impurity_calculator: ImpurityCalculator,
                        val loss_calculator: LossCalculator) extends Serializable {

  /**
   * Predict the value for the given data point using the model trained.
   *
   * @param fs features of the data point.
   * @return predicted value.
   */
  def predict(fs: SparseVector[Double]): Double = {
    DecisionTreeModel.predict(fs, root)
  }

  /**
   * Predict values for the given data using the model trained.
   *
   * @param input Array of [[bda.common.obj.LabeledPoint]] represent true label and features of data points
   * @return Array stored prediction
   */
  def predict(input: Seq[LabeledPoint]): Seq[Double] = {
    //val (pred, err) = computePredictAndError(input, 1.0)

    //Log.log("INFO", s"predict done, with mean error = ${err}")

    //pred

    computePredict(input, 1.0)
  }

  /**
   * Store decision tree model on the disk.
   *
   * @param path path of the location on the disk
   */
  def save(path: String): Unit = {
    DecisionTreeModel.save(path, this)
  }

  /**
   * Predict values for the given data using the model trained and the model weight
   *
   * @param input Array of [[bda.common.obj.LabeledPoint]] represent true label and features of data points
   * @param weight model weight
   * @return Array stored prediction.
   */
  def computePredict(input: Seq[LabeledPoint],
                             weight: Double): Seq[Double] = {

    val pred = input.map { lp =>
      val pred = DecisionTreeModel.predict(lp.fs, root) * weight
      pred
    }

    pred
  }

  /**
   * Update the pre-predictions for the given data using the model trained and the model weight
   *
   * @param input Array of [[bda.common.obj.LabeledPoint]] represent true label and features of data points
   * @param pre_pred pre-predictions
   * @param weight model weight
   * @return Array stored prediction.
   */
  def updatePredict(input: Seq[LabeledPoint],
                    pre_pred: Seq[Double],
                    weight: Double): Seq[Double] = {

    input.zip(pre_pred).map { case (lp, pre_pred) =>
      val pred = pre_pred + DecisionTreeModel.predict(lp.fs, root) * weight
      pred
    }
  }
}

object DecisionTreeModel {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point.
   * @param root root node of decision tree structure.
   * @return predicted value from the trained model.
   */
  def predict(fs: SparseVector[Double], root: DecisionTreeNode): Double = {
    var node = root
    while (!node.isLeaf) {
      if (fs(node.featureID) < node.splitValue) {
        node = node.leftNode.get
      } else {
        node = node.rightNode.get
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
 * Class of status of the node in a tree.
 *
 * @param impurity_calculator impurity calculator [[bda.local.model.tree.ImpurityCalculator]].
 * @param count number of instances the node has.
 * @param sum summation of labels of instances the node has.
 * @param squared_sum summation of squares of labels of instances the node has.
 * @param left_index leftmost id of instances the node has.
 * @param right_index next id of rightmost instances the node has.
 */
private[tree] class DecisionTreeStat(
            var impurity_calculator: ImpurityCalculator,
            var count: Int,
            var sum: Double,
            var squared_sum: Double,
            var left_index: Int,
            var right_index: Int) {

  /** information value of the node */
  var impurity = impurity_calculator.calculate(count, sum, squared_sum)

  override def toString: String = {
    s"count = $count, sum = $sum, sumSquares = $squared_sum, " +
      s"leftIndex = $left_index, rightIndex = $right_index, " +
      s"impurity = $impurity"
  }

  /**
   * Method to copy another stat.
   *
   * @param stat stat of another node
   */
  def copy(stat: DecisionTreeStat): Unit = {
    this.impurity_calculator = stat.impurity_calculator
    this.count = stat.count
    this.sum = stat.sum
    this.squared_sum = stat.squared_sum
    this.left_index = stat.left_index
    this.right_index = stat.right_index
    this.impurity = stat.impurity
  }

  /**
   * Method to udpate stat of the node with variations.
   *
   * @param count_bias count variation of the node
   * @param sum_bias sum variation of the node
   * @param sum_squares_bias sumSquares variation of the node
   * @param leftIndexBias left index variation of the node
   * @param rightIndexBias right index variation of the node
   */
  def update(
              count_bias: Int,
              sum_bias: Double,
              sum_squares_bias: Double,
              leftIndexBias: Int,
              rightIndexBias: Int): Unit = {

    count += count_bias
    sum += sum_bias
    squared_sum += sum_squares_bias
    left_index += leftIndexBias
    right_index += rightIndexBias
    impurity = impurity_calculator.calculate(count, sum, squared_sum)
  }

  /**
   * Method to add a instance which is next to the leftmost instance of the node.
   *
   * @param value the true label of the instance which will be added
   */
  def +=:(value: Double): Unit = {
    update(1, value, value * value, -1, 0)
  }

  /**
   * Method to add a instance which is next to the rightmost instance of the node.
   *
   * @param value the true label of the instance which will be added
   */
  def :+=(value: Double): Unit = {
    update(1, value, value * value, 0, 1)
  }

  /**
   * Method to subtract a instance which is next to the leftmost instance of the node.
   *
   * @param value the true label of the instance which will be subtracted
   */
  def -=:(value: Double): Unit = {
    update(-1, -1 * value, -1 * value * value, 1, 0)
  }

  /**
   * Method to subtract a instance which is next to the rightmost instance of the node.
   *
   * @param value the true label of the instance which will be subtracted
   */
  def :-=(value: Double): Unit = {
    update(-1, -1 * value, -1 * value * value, 0, -1)
  }
}

private[tree] object DecisionTreeStat {

  /**
   * Construct a [[bda.local.model.tree.DecisionTreeStat]] instance with original value.
   *
   * @return a [[bda.local.model.tree.DecisionTreeStat]] instance
   */
  def empty = {
    new DecisionTreeStat(VarianceCalculator, 0, 0, 0, 0, 0)
  }
}

/**
 * Class of nodes which form a tree.
 *
 * @param id integer node id, 1-based
 * @param dep node depth in a tree, 0-based
 * @param predict prediction of a leaf-node
 */
private[tree] class DecisionTreeNode(
            val id: Int,
            val dep: Int,
            val predict: Double) extends Serializable {

  /** flag to show whether is a leaf-node */
  var isLeaf = false
  /** threshold while splitting. Split left if feature < threshold, else right */
  var splitValue = 0.0
  /** feature index used in this splitting */
  var featureID = 0
  /** left child */
  var leftNode: Option[DecisionTreeNode] = None
  /** right child */
  var rightNode: Option[DecisionTreeNode] = None

  override def toString: String = {
    s"id = $id, dep = $dep, isLeaf = $isLeaf"
  }
}

private[tree] object DecisionTreeNode {

  /**
   * Construct a [[bda.local.model.tree.DecisionTreeNode]] instance with specified id, depth and prediction.
   *
   * @param nodeIndex node id, 1-based
   * @param nodeDep node depth in a tree, 0-based
   * @param predict prediction of a leaf-node
   * @return a [[bda.local.model.tree.DecisionTreeNode]] instance
   */
  def empty(nodeIndex: Int, nodeDep: Int, predict: Double): DecisionTreeNode = {
    new DecisionTreeNode(nodeIndex, nodeDep, predict)
  }

  /**
   * Calculate the id of the left child of this node.
   *
   * @param nodeIndex father node id
   * @return left child node id
   */
  def leftChildIndex(nodeIndex: Int): Int = {
    nodeIndex << 1
  }

  /**
   * Calculate the id of the right child of this node.
   * @param nodeIndex father node id
   * @return right child node id
   */
  def rightChildIndex(nodeIndex: Int): Int = {
    nodeIndex << 1 | 1
  }
}