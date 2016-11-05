package mymllib.spark.model.tree.gbrt

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import mymllib.spark.model.tree.TreeNode
import mymllib.spark.model.tree.cart.CARTModel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Model of GBRT(Gradient Boosting Regression Trees).
  *
  * @param impurity      impurity used to split node of model
  * @param max_depth     maximum depth of CART
  * @param max_bins      maximum number of bins
  * @param bin_samples   minimum number of samples used in finding splits and bins
  * @param min_node_size minimum number of data point instances in leaves
  * @param min_info_gain minimum information gain while splitting
  * @param num_round     number of rounds
  * @param learn_rate    learning rate
  * @param wk_learners   weak learners of GBRT model
  */
class GBRTModel(impurity: String,
                max_depth: Int,
                max_bins: Int,
                bin_samples: Int,
                min_node_size: Int,
                min_info_gain: Double,
                num_round: Int,
                learn_rate: Double,
                wk_learners: Array[TreeNode]) extends Serializable {

  /**
    * Method to predict value for given data point using the model trained.
    *
    * @param input test data set which represented as a RDD of [[LabeledPoint]]
    * @return a RDD stored predictions.
    */
  def predict(input: RDD[LabeledPoint]): RDD[(String, Double, Double)] = {
    val learn_rate = this.learn_rate
    val wk_learners = this.wk_learners

    input.map(lp => (lp.id, lp.label, GBRTModel.predict(lp.fs, wk_learners, learn_rate)))
  }

  /**
    * Predict value for the specified data point using the model trained.
    *
    * @param p test data point represented as an instance of [[SparseVector]]
    * @return the prediction for specified data point
    */
  def predict(p: SparseVector[Double]): Double = {
    GBRTModel.predict(p, wk_learners, learn_rate)
  }

  /**
    * Method to store model of GBRT on disk.
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
  * Static methods for [[GBRTModel]].
  */
object GBRTModel {
  /**
    * Method to load GBRT model from disk.
    *
    * @param sc Spark Context
    * @param fp path of GBDT model on disk
    * @return an instance of [[GBRTModel]]
    */
  def load(sc: SparkContext, fp: String): GBRTModel = {
    sc.objectFile[GBRTModel](fp).take(1)(0)
  }

  /**
    * Method to predict value for single data point using the model trained.
    *
    * @param fs         feature vector of single data point
    * @param learn_rate learning rate
    * @return the prediction for specified data point
    */
  def predict(fs: SparseVector[Double],
              wk_learners: Array[TreeNode],
              learn_rate: Double): Double = {
    require(0 < wk_learners.length,
      s"require length(weak learners) > 0 in GBRT model")

    var pred = CARTModel.predict(fs, wk_learners(0))
    Range(1, wk_learners.length).foreach {
      id =>
        pred += learn_rate * CARTModel.predict(fs, wk_learners(id))
    }
    pred
  }
}