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

  def generate_lchild(l_impurity: Double, l_pred: Double): Unit = {
    left_child = Some(Node.empty(id << 1, depth + 1))
    left_child.get.impurity = l_impurity
    left_child.get.predict = l_pred
  }

  def generate_rchild(r_impurity: Double, r_pred: Double): Unit = {
    right_child = Some(Node.empty(id << 1 | 1, depth + 1))
    right_child.get.impurity = r_impurity
    right_child.get.predict = r_pred
  }

}

object Node {

  def empty(id: Int, depth: Int): Node = {
    new Node(id, depth)
  }
}