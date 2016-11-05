package mymllib.spark.model.tree.cart

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import mymllib.spark.model.tree.TreeNode
import mymllib.spark.model.tree.cart.impurity.Impurity
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Model of CART(Classification And Regression Trees).
  *
  * @param root          root node of CART structure
  * @param n_fs          number of features
  * @param impurity      an instance of [[Impurity]]
  * @param max_depth     maximum depth of CART
  * @param max_bins      maximum number of bins
  * @param bin_samples   minimum number of samples used in finding splits and bins
  * @param min_node_size minimum number of data point instances in leaves
  * @param min_info_gain minimum information gain while splitting
  * @param row_rate      sampling ratio of training data set
  * @param col_rate      sampling ratio of features
  */
class CARTModel(val root: TreeNode,
                val n_fs: Int,
                val impurity: Impurity,
                val max_depth: Int,
                val max_bins: Int,
                val bin_samples: Int,
                val min_node_size: Int,
                val min_info_gain: Double,
                val row_rate: Double,
                val col_rate: Double) extends Serializable {

  /**
    * Predict values for the given data using the model trained.
    *
    * @param input test data set which represented as a RDD of [[LabeledPoint]]
    * @return a RDD stored predictions.
    */
  def predict(input: RDD[LabeledPoint]): RDD[(String, Double, Double)] = {
    val root = this.root
    input.map(lp => (lp.id, lp.label, CARTModel.predict(lp.fs, root)))
  }

  /**
    * Predict value for the specified data point using the model trained.
    *
    * @param p test data point represented as an instance of [[SparseVector]]
    * @return the prediction for specified data point
    */
  def predict(p: SparseVector[Double]): Double = {
    CARTModel.predict(p, root)
  }

  /**
    * Method to store model of CART on disk.
    *
    * @param sc an instance of [[SparkContext]]
    * @param pt path of the model location on disk
    */
  def save(sc: SparkContext, pt: String): Unit = {
    val model_rdd: RDD[CARTModel] = sc.makeRDD(Seq(this))
    model_rdd.saveAsObjectFile(pt)
  }

  /**
    * Method to print structure of CART model.
    */
  def printStructure() = {
    CARTModel.printStructure(this.root)
  }
}

/**
  * Static methods for [[CARTModel]].
  */
object CARTModel {

  /**
    * Method to predict value for single data point using the model trained.
    *
    * @param fs feature vector of single data point
    * @param root root of CART model
    * @return the prediction for specified data point
    */
  private[tree] def predict(fs: SparseVector[Double], root: TreeNode): Double = {
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

  /**
    * Method to load CART model from disk.
    *
    * @param sc Spark Context
    * @param fp path of CART model on disk
    * @return an instance of [[CARTModel]]
    */
  def load(sc: SparkContext, fp: String): CARTModel = {
    sc.objectFile[CARTModel](fp).take(1)(0)
  }

  /**
    * Method to print structure of CART model.
    *
    * @param root root of CART model
    */
  def printStructure(root: TreeNode): Unit = {
    val prefix = Array.fill[String](root.depth)("|---").mkString("")
    println(s"$prefix$root")

    root.left_child match {
      case Some(l_child: TreeNode) =>
        printStructure(l_child)
      case None => // RETURN
    }
    root.right_child match {
      case Some(r_child: TreeNode) =>
        printStructure(r_child)
      case None => // RETURN
    }
  }
}