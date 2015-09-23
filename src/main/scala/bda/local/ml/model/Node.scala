package bda.local.ml.model

class Node(
    val id: Int,
    val dep: Int,
    val predict: Double) {

  var isLeaf = false
  var splitValue = 0.0
  var featureID = 0
  var leftNode: Option[Node] = None
  var rightNode: Option[Node] = None

  override def toString: String = {
    s"id = $id, dep = $dep, isLeaf = $isLeaf"
  }
}

object Node {

  def empty(nodeIndex: Int, nodeDep: Int, predict: Double): Node = {
    new Node(nodeIndex, nodeDep, predict)
  }

  def leftChildIndex(nodeIndex: Int): Int = {
    nodeIndex << 1
  }

  def rightChildIndex(nodeIndex: Int): Int = {
    nodeIndex << 1 | 1
  }
}