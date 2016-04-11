package bda.spark.model.tree.cart

import bda.common.util.Sampler
import bda.spark.model.tree.{NodeBestSplit, FeatureSplit}

/**
  * Class of nodes which form a CART model.
  *
  * @param id ID of the node, 1-based
  * @param depth depth of the node in a CART model, 0-based
  * @param n_fs number of features
  * @param col_rate sampling ratio of features
  * @param impurity impurity value of the node
  * @param predict prediction value of the node
  */
private[cart] class CARTNode (val id: Int,
                              val depth: Int,
                              val n_fs: Int,
                              val col_rate: Double,
                              val impurity: Double,
                              val predict: Double) extends Serializable {

  /** flag to show whether is a leaf-node */
  var is_leaf: Boolean = true
  /** splitting way of this node */
  var split: Option[FeatureSplit] = None
  /** left child */
  var left_child: Option[CARTNode] = None
  /** right child */
  var right_child: Option[CARTNode] = None
  /** sub features */
  val sub_fs = Sampler.subSample(n_fs, col_rate)

  /**
    * Method to convert the node into a [[String]].
    *
    * @return a instance of [[String]] represented the node
    */
  override def toString = {
    s"id($id), depth($depth), impurity($impurity), predict($predict), " +
      s"split(${split.getOrElse("NoSplit")})"//, sub_fs(${sub_fs.mkString(",")})"
  }

  /**
    * Method to split the node if satisfies conditions:
    *     1.  information gain >= params.min_info_gain
    *     2.  depth < params.max_depth
    *     3.  labels count of left child >= params.min_node_size
    *     4.  labels count of right child >= params.min_node_size
    *
    * @param best_split best split of the node
    * @param max_depth maximum of depth
    * @param min_info_gain minimum information gain while splitting
    * @param min_node_size minimum size of leaves
    */
  def split(best_split: NodeBestSplit,
            max_depth: Int,
            min_info_gain: Double,
            min_node_size: Int): Unit = {

    val info_gain = impurity - best_split.weight_impurity

    if ((info_gain >= min_info_gain)
      && (depth < max_depth)
      && (best_split.l_count >= min_node_size)
      && (best_split.r_count >= min_node_size)) {

      is_leaf = false
      split = Some(best_split.split)
      left_child = Some(new CARTNode(id << 1,
        depth + 1,
        n_fs,
        col_rate,
        best_split.l_impurity,
        best_split.l_predict))
      right_child = Some(new CARTNode((id << 1) + 1,
        depth + 1,
        n_fs,
        col_rate,
        best_split.r_impurity,
        best_split.r_predict))
    }
  }
}
