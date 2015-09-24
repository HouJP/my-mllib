package bda.local.ml.model

/**
 * Class of nodes which form a tree.
 *
 * @param id integer node id, 1-based
 * @param dep node depth in a tree, 0-based
 * @param predict prediction of a leaf-node
 */
class Node(
    val id: Int,
    val dep: Int,
    val predict: Double) {

  /** flag to show whether is a leaf-node */
  var isLeaf = false
  /** threshold while splitting. Split left if feature < threshold, else right */
  var splitValue = 0.0
  /** feature index used in this splitting */
  var featureID = 0
  /** left child */
  var leftNode: Option[Node] = None
  /** right child */
  var rightNode: Option[Node] = None

  override def toString: String = {
    s"id = $id, dep = $dep, isLeaf = $isLeaf"
  }
}

object Node {

  /**
   * Construct a [[Node]] instance with specified id, depth and prediction.
   *
   * @param nodeIndex node id, 1-based
   * @param nodeDep node depth in a tree, 0-based
   * @param predict prediction of a leaf-node
   * @return a [[Node]] instance
   */
  def empty(nodeIndex: Int, nodeDep: Int, predict: Double): Node = {
    new Node(nodeIndex, nodeDep, predict)
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