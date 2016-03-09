package bda.spark.model.tree.cart

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import bda.spark.model.tree.cart.impurity.Impurity
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class CARTModel(root: CARTNode,
                n_fs: Int,
                impurity: Impurity,
                max_depth: Int,
                max_bins: Int,
                bin_samples: Int,
                min_node_size: Int,
                min_info_gain: Double,
                row_rate: Double,
                col_rate: Double) extends Serializable {

  def predict(input: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val root = this.root
    input.map(lp => (lp.label, CARTModel.predict(lp.fs, root)))
  }

  def save(sc: SparkContext, pt: String): Unit = {
    val model_rdd = sc.makeRDD(Seq(this))
    model_rdd.saveAsObjectFile(pt)
  }

  def printStructure = {
    CARTModel.printStructure(this.root)
  }
}

object CARTModel {

  private[cart] def predict(fs: SparseVector[Double], root: CARTNode): Double = {
    var node = root
    while (!node.is_leaf) {
      if (fs(node.split.get.id_f) < node.split.get.threshold) {
        node = node.left_child.get
      } else {
        node = node.right_child.get
      }
    }
    node.predict
  }

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