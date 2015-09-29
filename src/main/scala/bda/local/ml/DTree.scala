package bda.local.ml

import bda.local.ml.model.{LabeledPoint, DTreeModel, Stat, Node}
import bda.local.ml.para.DTreePara
import bda.local.ml.util.Log
import scala.collection.mutable

/**
 * A class which implements a decision tree algorithm for classification and regression.
 *
 * @param dt_para the configuration parameters for the decision tree algorithm
 */
class DTree (private val dt_para: DTreePara) {

  /**
   * Method to train a decision tree over a training data which represented as an array of [[bda.local.ml.model.LabeledPoint]]
   *
   * @param input traning data: Array of [[bda.local.ml.model.LabeledPoint]]
   * @return a [[bda.local.ml.model.DTreeModel]] instance which can be used for prediction
   */
  def fit(input: Array[LabeledPoint]): DTreeModel = {
    val num_f = input(0).features.size
    val num_d = input.length

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
      val value = input(index).label
      sum += value
      sum_squares += value * value
    }
    val stat = new Stat(
      dt_para.impurity,
      count,
      sum,
      sum_squares,
      index_l,
      index_r) // nodeinfo

    // root of the decision tree
    val pre = dt_para.loss_calculator.predict(stat)
    val root = Node.empty(nodeIndex = 1, nodeDep = 0, predict = pre)

    val que_node = new mutable.Queue[(Node, Stat)]
    que_node.enqueue((root, stat)) // topNode covers dataIndex [0, numData)

    while (que_node.nonEmpty) {
      findBestSplit(input, index_d, que_node, num_f)
    }

    new DTreeModel(root, dt_para)
  }

  /**
   * Check the node whether or not splits
   * @param node the node waiting to split
   * @param stat the node status
   * @return whether or not splits
   */
  def checkNode(node: Node, stat: Stat): Boolean = {
    if (node.dep >= dt_para.max_depth) {
      return false
    }
    if (stat.count <= dt_para.min_node_size) {
      return false
    }
    return true
  }

  /**
   * Find the best threshold to split the front node in the queue.
   *
   * @param input training data
   * @param index_d the index array of the training data
   * @param que_node nodes queue which stored the nodes need to be splited
   * @param num_f training data metadata and the strategy of the decision tree
   */
  def findBestSplit(
      input: Array[LabeledPoint],
      index_d: Array[Int],
      que_node: mutable.Queue[(Node, Stat)],
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
    var max_ig = dt_para.min_info_gain
    var best_stat_l = Stat.empty
    var best_stat_r = Stat.empty
    for (featureIndex <- 0 until num_f) {
      var a_best_sp_v = 0.0
      var a_max_ig = max_ig
      var a_best_stat_l: Stat = Stat.empty
      var a_best_stat_r: Stat = Stat.empty

      val stat_l = new Stat(
        dt_para.impurity,
        0,
        0,
        0,
        stat.leftIndex,
        stat.leftIndex)
      val stat_r = new Stat(
        dt_para.impurity,
        stat.count,
        stat.sum,
        stat.sumSquares,
        stat.leftIndex,
        stat.rightIndex)

      val f_v = new Array[Double](stat.count)
      val index_tmp = new Array[Int](stat.count)
      for (offset <- 0 until stat.count) {
        f_v(offset) = input(index_d(stat.leftIndex + offset)).features(featureIndex)
        index_tmp(offset) = index_d(stat.leftIndex + offset)
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
          index_d(stat.leftIndex + dataOffset) = ordered(dataOffset)._1
        }
      }
    }

    // add leftChild and rightChild to the queue
    if (max_ig > dt_para.min_info_gain) {
      val ind_l = Node.leftChildIndex(node.id)
      val ind_r = Node.rightChildIndex(node.id)
      val pre_l = dt_para.loss_calculator.predict(best_stat_l)
      val pre_r = dt_para.loss_calculator.predict(best_stat_r)
      val node_l = Node.empty(ind_l, node.dep + 1, pre_l)
      val node_r = Node.empty(ind_r, node.dep + 1, pre_r)

      node.splitValue = best_sp_v
      node.featureID = best_f_id
      node.leftNode = Option(node_l)
      node.rightNode = Option(node_r)

      que_node.enqueue((node_l, best_stat_l))
      que_node.enqueue((node_r, best_stat_r))

      //Log.log("INFO", s"node_${node.id} split into node_${leftNode.id} and node_${rightNode.id}, " +
      //  s"with splitValue = ${node.splitValue} and featureID = ${node.featureID}")
      //Log.log("INFO", s"\t\tnode_${node.id}'s stat: $stat")
      //Log.log("INFO", s"\t\tnode_${leftNode.id}'s stat: $bestLeftStat")
      //Log.log("INFO", s"\t\tnode_${rightNode.id}'s stat: $bestRightStat")
    } else {
      node.isLeaf = true

      //Log.log("INFO", s"node_${node.id} stop split, with predict = ${node.predict}")
    }
  }
}