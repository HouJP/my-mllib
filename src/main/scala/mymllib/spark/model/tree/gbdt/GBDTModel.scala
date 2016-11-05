package mymllib.spark.model.tree.gbdt

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import mymllib.spark.model.tree.TreeNode
import mymllib.spark.model.tree.gbdt.impurity.Impurity
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Model of GBDT(Gradient Boosting Decision Trees).
  *
  * @param impurity      impurity used to split node of model
  * @param max_depth     maximum depth of CART
  * @param max_bins      maximum number of bins
  * @param bin_samples   minimum number of samples used in finding splits and bins
  * @param min_node_size minimum number of data point instances in leaves
  * @param min_info_gain minimum information gain while splitting
  * @param num_round     number of rounds
  * @param num_label     number of different labels
  * @param wk_learners   weak learners of GBDT model
  */
class GBDTModel(impurity: Impurity,
                max_depth: Int,
                max_bins: Int,
                bin_samples: Int,
                min_node_size: Int,
                min_info_gain: Double,
                num_round: Int,
                num_label: Int,
                wk_learners: Array[TreeNode]) extends Serializable{

  /**
    * Method to predict value for given data point using the model trained.
    *
    * @param input test data set which represented as a RDD of [[LabeledPoint]]
    * @return a RDD stored predictions.
    */
  def predict(input: RDD[LabeledPoint]): RDD[(String, Double, Double)] = {
    val wk_learners = this.wk_learners
    val num_label = this.num_label

    input.map(lp => (lp.id, lp.label, GBDTModel.predict(lp.fs, wk_learners, num_label)))
  }

  /**
    * Predict value for the specified data point using the model trained.
    *
    * @param p test data point represented as an instance of [[SparseVector]]
    * @return the prediction for specified data point
    */
  def predict(p: SparseVector[Double]): Double = {
    GBDTModel.predict(p, wk_learners, num_label)
  }

  /**
    * Method to store model of GBDT on disk.
    *
    * @param sc an instance of [[SparkContext]]
    * @param pt path of the model location on disk
    */
  def save(sc: SparkContext, pt: String): Unit = {
    val model_rdd = sc.makeRDD(Seq(this))
    model_rdd.saveAsObjectFile(pt)
  }
}

/**
  * Static methods for [[GBDTModel]].
  */
object GBDTModel {

  /**
    * Method to predict value for single data point using the model trained.
    *
    * @param fs      feature vector of single data point
    * @param n_label number of different labels
    * @return the prediction for specified data point
    */
  private[gbdt] def predict(fs: SparseVector[Double],
              wk_learners: Array[TreeNode],
              n_label: Int): Double = {
    val preds = Array.fill[Double](n_label)(0.0)

    wk_learners.indices.foreach {
      id =>
        preds(id % n_label) += GBDTModel.predict(fs, wk_learners(id))
    }

    preds.zipWithIndex.maxBy(_._1)._2
  }

  /**
    * Method to predict value for single data point using the model trained.
    *
    * @param fs   feature vector of single data point
    * @param root root of CART model
    * @return the prediction for specified data point
    */
  private[gbdt] def predict(fs: SparseVector[Double], root: TreeNode): Double = {
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
    * Method to load GBDT model from disk.
    *
    * @param sc Spark Context
    * @param fp path of GBDT model on disk
    * @return an instance of [[GBDTModel]]
    */
  def load(sc: SparkContext, fp: String): GBDTModel = {
    sc.objectFile[GBDTModel](fp).take(1)(0)
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