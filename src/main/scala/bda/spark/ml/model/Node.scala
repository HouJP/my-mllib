package bda.spark.ml.model

class Node (
    val id: Int,
    val depth: Int) extends Serializable {

  var is_leaf: Boolean = true
  var impurity: Double = 0.0
  var predict: Double = 0.0
  var split: Option[Split] = None
  var left_child: Option[Node] = None
  var right_child: Option[Node] = None



  override def toString: String = {
    s"Node: id = $id, predict = $predict, impurity = $impurity"
  }
}

object Node {

  def empty(id: Int, depth: Int): Node = {
    new Node(id, depth)
  }

  def generate_lchild(node: Node): Node = {
    empty(node.id << 1, node.depth + 1)
  }

  def generate_rchild(node: Node): Node = {
    empty(node.id << 1 | 1, node.depth + 1)
  }
}