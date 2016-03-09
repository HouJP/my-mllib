package bda.spark.model.tree.cart

object CARTModel {

  def printStructure(root: CARTNode): Unit = {
    val prefix = Array.fill[String](root.depth)("|---").mkString("")
    println(s"$prefix$root")

    root.left_child match {
      case Some(l_child: CARTNode) =>
        printStructure(l_child)
      case None => // RETURN
    }
    root.right_child match {
      case Some(r_child: CARTNode) =>
        printStructure(r_child)
      case None => // RETURN
    }
  }
}