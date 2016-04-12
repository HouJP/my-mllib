package bda.spark.model.tree.rf

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import bda.spark.model.tree.TreeNode
import bda.spark.model.tree.cart.CARTModel
import bda.spark.model.tree.rf.impurity.{Impurity, Impurities}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Model of Random Forest.
  *
  * @param wk_learners   weak learners of Random Forest model
  * @param impurity      impurity used to split node of model
  * @param max_depth     maximum depth of CART
  * @param max_bins      maximum number of bins
  * @param bin_samples   minimum number of samples used in finding splits and bins
  * @param min_node_size minimum number of data point instances in leaves
  * @param min_info_gain minimum information gain while splitting
  * @param row_rate sample ratio of train data set
  * @param col_rate sample ratio of features
  * @param num_trees Number of decision trees
  */
class RandomForestModel(wk_learners: Array[TreeNode],
                        impurity: String,
                        max_depth: Int,
                        max_bins: Int,
                        bin_samples: Int,
                        min_node_size: Int,
                        min_info_gain: Double,
                        row_rate: Double,
                        col_rate: Double,
                        num_trees: Int) {

  /**
    * Method to predict value for given data point using the model trained.
    *
    * @param input test data set which represented as a RDD of [[LabeledPoint]]

    * @return a RDD stored predictions.
    */
  def predict(input: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val impurity = Impurities.fromString(this.impurity)
    val wk_learners = this.wk_learners

    input.map(lp => (lp.label, RandomForestModel.predict(lp.fs, impurity, wk_learners)))
  }

  /**
    * Predict value for the specified data point using the model trained.
    *
    * @param p test data point represented as an instance of [[SparseVector]]
    * @return the prediction for specified data point
    */
  def predict(p: SparseVector[Double]): Double = {
    val impurity = Impurities.fromString(this.impurity)
    RandomForestModel.predict(p, impurity, wk_learners)
  }

  /**
    * Method to store model of Random Forest on disk.
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
  * Static methods for [[RandomForestModel]].
  */
private[rf] object RandomForestModel {

  /**
    * Method to predict value for single data point using the model trained.
    *
    * @param fs          feature vector of single data point
    * @param impurity    impurity used to split node
    * @param wk_learners weak learners of the model
    * @return the prediction for specified data point
    */
  def predict(fs: SparseVector[Double],
              impurity: Impurity,
              wk_learners: Array[TreeNode]): Double = {
    require(0 < wk_learners.length,
      s"require length(weak learners) > 0 in Random Forest model")

    var pred = CARTModel.predict(fs, wk_learners(0))
    impurity.predict(fs, wk_learners)
  }
}