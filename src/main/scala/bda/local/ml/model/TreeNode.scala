package bda.local.ml.model

class TreeNode (
    val id: Int,
    var leftChild: Option[TreeNode] = None,
    var rightChild: Option[TreeNode] = None) {

  override def toString: String = {
    s"id = $id, " +
      s"leftChildID = ${leftChild.getOrElse(new TreeNode(-1).id)}, " +
      s"rightChildID = ${rightChild.getOrElse(new TreeNode(-1)).id}"
  }
}