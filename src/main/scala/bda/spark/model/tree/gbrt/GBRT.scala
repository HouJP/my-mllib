package bda.spark.model.tree.gbrt

import bda.common.Logging
import bda.common.obj.LabeledPoint
import bda.common.util.{Timer, Msg}
import bda.spark.model.tree.TreeNode
import bda.spark.model.tree.cart.{CARTModel, CART}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * External interface of GBRT(Gradient Boosting Regression Trees) on spark.
  * Reference:
  *    Friedman, J. H. (2000). Greedy Function Approximation : A Gradient Boosting Machine. Annals of Statistics.
  */
object GBRT extends Logging {

  /**
    * An adapter for training a GBRT model.
    *
    * @param train_data    training data set
    * @param impurity      impurity used to split node, default is "Variance"
    * @param max_depth     maximum depth of the CART default is 10
    * @param max_bins      maximum number of bins, default is 32
    * @param bin_samples   minimum number of samples used to find [[bda.spark.model.tree.FeatureSplit]]
    *                      and [[bda.spark.model.tree.FeatureBin]], default is 10000
    * @param min_node_size minimum number of instances in leaves, default is 15
    * @param min_info_gain minimum infomation gain while splitting, default is 1e-6
    * @param num_round     number of rounds for GBDT
    * @param learn_rate    learning rate of iteration
    * @return an instance of [[GBRTModel]]
    */
  def train(train_data: RDD[LabeledPoint],
            impurity: String = "Variance",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            num_round: Int = 10,
            learn_rate: Double = 0.02): GBRTModel = {

    val msg = Msg("n(train_data)" -> train_data.count(),
      "impurity" -> impurity,
      "max_depth" -> max_depth,
      "max_bins" -> max_bins,
      "bin_samples" -> bin_samples,
      "min_node_size" -> min_node_size,
      "min_info_gain" -> min_info_gain,
      "num_round" -> num_round,
      "learn_rate" -> learn_rate
    )
    logInfo(msg.toString)

    new GBRT(impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      num_round,
      learn_rate).train(train_data)
  }
}

/**
  * Class of GBRT(Gradient Boosting Regression Tree).
  *
  * @param impurity      impurity used to split node
  * @param max_depth     maximum depth of CART
  * @param max_bins      maximum number of bins
  * @param bin_samples   minimum number of samples used to find [[bda.spark.model.tree.FeatureSplit]] and [[bda.spark.model.tree.FeatureBin]]
  * @param min_node_size minimum number of instances in leaves
  * @param min_info_gain minimum information gain while splitting
  * @param num_round     number of rounds for GBDT
  * @param learn_rate    learning rate of iteration
  */
class GBRT(impurity: String,
           max_depth: Int,
           max_bins: Int,
           bin_samples: Int,
           min_node_size: Int,
           min_info_gain: Double,
           num_round: Int,
           learn_rate: Double) extends Logging {

  /**
    * Method to train a GBRT model based on training data set.
    *
    * @param train_data training data set represented as a RDD of [[LabeledPoint]]
    * @return an instance of [[GBRTModel]]
    */
  def train(train_data: RDD[LabeledPoint]): GBRTModel = {
    val timer = new Timer()

    // Build container for roots
    val wk_learners = mutable.ArrayBuffer[TreeNode]()
    // var wk_learners = new Array[TreeNode](0)

    // Convert LabeledPoint to GBRTPoint
    var gbrt_ps = GBRTPoint.toGBRTPoint(train_data).persist()
    // Convert GBRTPoint to training data set of CART
    val cart_ps = gbrt_ps.map {
      p =>
        LabeledPoint(p.label - p.f, p.fs)
    }

    // Build weak learner #0
    val wl0 = new CART(impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      1.0, 1.0).train(cart_ps)
    wk_learners += wl0.root

    logInfo("GBRT Model round#1 training done")

    var iter = 1
    while (iter < num_round) {
      val factor = if (1 == iter) {
        1.0
      } else {
        learn_rate
      }
      // Update GBRT points
      val pre_gbrt_ps = gbrt_ps
      gbrt_ps = gbrt_ps.map {
        p =>
          val new_f = p.f + factor * CARTModel.predict(p.fs, wk_learners.last)
          GBRTPoint(p.label, new_f, p.fs)
      }.persist()
      // Check point
      if (19 == iter % 20) {
        gbrt_ps.checkpoint()
      }
      gbrt_ps.count()
      pre_gbrt_ps.unpersist()
      // Update CART points
      val cart_ps = gbrt_ps.map {
        p =>
          LabeledPoint(p.label - p.f, p.fs)
      }
      // Build weak learner #iter
      val wl = new CART(impurity,
        max_depth,
        max_bins,
        bin_samples,
        min_node_size,
        min_info_gain,
        1.0, 1.0).train(cart_ps)
      wk_learners += wl.root

      logInfo(s"GBRT Model round#${iter + 1} training done")
      iter += 1
    }

    logInfo(s"GBRT Model training done, cost time ${timer.cost()}ms")

    new GBRTModel(this.impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      num_round,
      learn_rate,
      wk_learners.toArray)
  }
}