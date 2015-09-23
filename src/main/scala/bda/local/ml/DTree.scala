package bda.local.ml

import bda.local.ml.model._
import bda.local.ml.util.Log
import scala.collection.mutable

class DTree (private val dTreetrategy: DTreeStrategy) {

  def run(input: Array[LabeledPoint]): DTreeModel = {
    val dTreeMetadata = DTreeMetadata.build(input, dTreetrategy)

    val dataIndex = new Array[Int](dTreeMetadata.numData)
    for (index <- 0 until dTreeMetadata.numData) {
      dataIndex(index) = index
    }

    // stat of the root
    val leftIndex = 0
    val rightIndex = dTreeMetadata.numData // topNode covers data from 0 until numData: [0, numData)
    val count = rightIndex - leftIndex
    var sum = 0.0
    var sumSquares = 0.0
    for (index <- leftIndex until rightIndex) {
      val value = input(index).label
      sum += value
      sumSquares += value * value
    }
    val stat = new Stat(dTreetrategy.impurity, count, sum, sumSquares, leftIndex, rightIndex)

    // root of the decision tree
    val predict = dTreetrategy.loss.predict(stat)
    val topNode = Node.empty(nodeIndex = 1, nodeDep = 0, predict = predict)

    val nodeQueue = new mutable.Queue[(Node, Stat)]
    nodeQueue.enqueue((topNode, stat)) // topNode covers dataIndex [0, numData)

    while (nodeQueue.nonEmpty) {
      DTree.findBestSplit(input, dataIndex, nodeQueue, dTreeMetadata)
    }

    new DTreeModel(topNode, dTreetrategy)
  }
}

object DTree {

  def train(input: Array[LabeledPoint], dTreeStrategy: DTreeStrategy): DTreeModel = {
    new DTree(dTreeStrategy).run(input)
  }

  def checkNode(node: Node, stat: Stat, dTreeMetadata: DTreeMetadata): Boolean = {
    if (node.dep >= dTreeMetadata.dTreeStrategy.maxDepth) {
      return false
    }
    if (stat.count <= dTreeMetadata.dTreeStrategy.minNodeSize) {
      return false
    }
    return true
  }

  def findBestSplit(
      input: Array[LabeledPoint],
      dataIndex: Array[Int],
      nodeQueue: mutable.Queue[(Node, Stat)],
      dTreeMetadata: DTreeMetadata): Unit = {

    val (node, stat) = nodeQueue.dequeue()

    // check node, split it if necessary
    if (!checkNode(node, stat, dTreeMetadata)) {
      node.isLeaf = true

      Log.log("INFO", s"node_${node.id} stop split")
      return
    }

    // find the best split
    var bestSplitValue = 0.0
    var bestFeatureID = 0
    var maxInfoGain = dTreeMetadata.dTreeStrategy.minInfoGain
    var bestLeftStat = Stat.empty
    var bestRightStat = Stat.empty
    for (featureIndex <- 0 until dTreeMetadata.numFeatures) {
      var aMaxInfoGain = maxInfoGain
      var aBestSplitValue = 0.0
      var aBestLeftStat: Stat = Stat.empty
      var aBestRightStat: Stat = Stat.empty

      var leftStat = new Stat(dTreeMetadata.dTreeStrategy.impurity, 0, 0, 0, stat.leftIndex, stat.leftIndex)
      var rightStat = new Stat(dTreeMetadata.dTreeStrategy.impurity, stat.count, stat.sum, stat.sumSquares, stat.leftIndex, stat.rightIndex)

      val fValue = new Array[Double](stat.count)
      val tmpIndex = new Array[Int](stat.count)
      for (dataOffset <- 0 until stat.count) {
        fValue(dataOffset) = input(dataIndex(stat.leftIndex + dataOffset)).features(featureIndex)
        tmpIndex(dataOffset) = dataIndex(stat.leftIndex + dataOffset)
      }
      val orderedArr = tmpIndex.zip(fValue).sortBy(_._2)

      for (dataOffset <- 0 until (stat.count - 1)) {
        val dataValue = input(orderedArr(dataOffset)._1).label
        leftStat :+ dataValue
        dataValue -: rightStat

        if (orderedArr(dataOffset + 1)._2 > orderedArr(dataOffset)._2) {
          val splitValue = (orderedArr(dataOffset + 1)._2 + orderedArr(dataOffset)._2) / 2
          val infoGain = stat.impurity - (1.0 * leftStat.count / stat.count) * leftStat.impurity - (1.0 * rightStat.count / stat.count) * rightStat.impurity

          if (infoGain > aMaxInfoGain) {
            aMaxInfoGain = infoGain
            aBestSplitValue = splitValue
            aBestLeftStat.copy(leftStat)
            aBestRightStat.copy(rightStat)
          }
        }
      }

      if (aMaxInfoGain > maxInfoGain) {
        maxInfoGain = aMaxInfoGain
        bestSplitValue = aBestSplitValue
        bestFeatureID = featureIndex
        bestLeftStat.copy(aBestLeftStat)
        bestRightStat.copy(aBestRightStat)

        for (dataOffset <- 0 until stat.count) {
          dataIndex(stat.leftIndex + dataOffset) = orderedArr(dataOffset)._1
        }
      }
    }

    // add leftChild and rightChild to the queue
    if (maxInfoGain > dTreeMetadata.dTreeStrategy.minInfoGain) {
      val leftIndex = Node.leftChildIndex(node.id)
      val rightIndex = Node.rightChildIndex(node.id)
      val leftPre = dTreeMetadata.dTreeStrategy.loss.predict(bestLeftStat)
      val rightPre = dTreeMetadata.dTreeStrategy.loss.predict(bestRightStat)
      val leftNode = Node.empty(leftIndex, node.dep + 1, leftPre)
      val rightNode = Node.empty(rightIndex, node.dep + 1, rightPre)

      node.splitValue = bestSplitValue
      node.featureID = bestFeatureID
      node.leftNode = Option(leftNode)
      node.rightNode = Option(rightNode)

      nodeQueue.enqueue((leftNode, bestLeftStat))
      nodeQueue.enqueue((rightNode, bestRightStat))

      Log.log("INFO", s"node_${node.id} split into node_${leftNode.id} and node_${rightNode.id}, " +
        s"with splitValue = ${node.splitValue} and featureID = ${node.featureID}")
      Log.log("INFO", s"\t\tnode_${node.id}'s stat: $stat")
      Log.log("INFO", s"\t\tnode_${leftNode.id}'s stat: $bestLeftStat")
      Log.log("INFO", s"\t\tnode_${rightNode.id}'s stat: $bestRightStat")
    } else {
      node.isLeaf = true

      Log.log("INFO", s"node_${node.id} stop split, with predict = ${node.predict}")
    }
  }
}