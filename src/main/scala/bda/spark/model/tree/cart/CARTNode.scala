package bda.spark.model.tree.cart

import bda.common.util.Sampler

private[cart] class CARTNode (val id: Int,
                              val depth: Int,
                              val n_fs: Int,
                              val col_rate: Double,
                              val impurity: Double,
                              val predict: Double) extends Serializable {

  /** flag to show whether is a leaf-node */
  var is_leaf: Boolean = true
  /** splitting way of this node */
  var split: Option[CARTSplit] = None
  /** left child */
  var left_child: Option[CARTNode] = None
  /** right child */
  var right_child: Option[CARTNode] = None
  /** sub features */
  val sub_fs = Sampler.subSample(n_fs, col_rate)

  override def toString = {
    s"id($id), depth($depth), impurity($impurity), predict($predict), " +
      s"split(${split.getOrElse("NoSplit")}), sub_fs(${sub_fs.mkString(",")})"
  }

  def split(best_split: CARTBestSplit,
            max_depth: Int,
            min_info_gain: Double,
            min_node_size: Int): Unit = {

    val info_gain = impurity - best_split.weight_impurity

    println("Into Split ...")

    if ((info_gain >= min_info_gain)
      && (depth < max_depth)
      && (best_split.l_count >= min_node_size)
      && (best_split.r_count >= min_node_size)) {

      println("Into grow LR child ...")

      is_leaf = false
      split = Some(best_split.split)
      left_child = Some(new CARTNode(id << 1,
        depth + 1,
        n_fs,
        col_rate,
        best_split.l_impurity,
        best_split.l_count))
      right_child = Some(new CARTNode((id << 1) + 1,
        depth + 1,
        n_fs,
        col_rate,
        best_split.r_impurity,
        best_split.r_count))
    }
  }
}

private[cart] object CARTNode {

}