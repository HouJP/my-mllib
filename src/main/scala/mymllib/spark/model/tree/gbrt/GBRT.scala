package mymllib.spark.model.tree.gbrt

import bda.common.Logging
import bda.common.obj.LabeledPoint
import bda.common.util.{Timer, Msg}
import mymllib.spark.model.tree.TreeNode
import mymllib.spark.model.tree.cart.{CARTModel, CART}
import org.apache.spark.rdd.RDD
import mymllib.spark.evaluate.Regression._

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
    * @param watchlist     specify validations set to watch performance
    * @param impurity      impurity used to split node, default is "Variance"
    * @param max_depth     maximum depth of the CART default is 10
    * @param max_bins      maximum number of bins, default is 32
    * @param bin_samples   minimum number of samples used to find [[mymllib.spark.model.tree.FeatureSplit]]
    *                      and [[mymllib.spark.model.tree.FeatureBin]], default is 10000
    * @param min_node_size minimum number of instances in leaves, default is 15
    * @param min_info_gain minimum infomation gain while splitting, default is 1e-6
    * @param num_round     number of rounds for GBDT
    * @param learn_rate    learning rate of iteration
    * @return an instance of [[GBRTModel]]
    */
  def train(train_data: RDD[LabeledPoint],
            watchlist: Array[(String, RDD[LabeledPoint])],
            impurity: String = "Variance",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            num_round: Int = 10,
            learn_rate: Double = 0.02): GBRTModel = {

    val msg = Msg("n(train_data)" -> train_data.count(),
      "watchlist" -> s"[${watchlist.map(_._1).mkString(",")}]",
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
      learn_rate).train(train_data, watchlist)
  }
}

/**
  * Class of GBRT(Gradient Boosting Regression Tree).
  *
  * @param impurity      impurity used to split node
  * @param max_depth     maximum depth of CART
  * @param max_bins      maximum number of bins
  * @param bin_samples   minimum number of samples used to find [[mymllib.spark.model.tree.FeatureSplit]] and [[mymllib.spark.model.tree.FeatureBin]]
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
    * @param watchlist  specify validations set to watch performance
    * @return an instance of [[GBRTModel]]
    */
  def train(train_data: RDD[LabeledPoint], watchlist: Array[(String, RDD[LabeledPoint])]): GBRTModel = {
    val timer = new Timer()
    val learn_rate = this.learn_rate
    // Build container for roots
    val wk_learners = mutable.ArrayBuffer[TreeNode]()
    // Convert LabeledPoint to GBRTPoint
    var gbrt_ps = GBRTPoint.toGBRTPoint(train_data).persist()
    // Convert GBRTPoint to training data set of CART
    val cart_ps = train_data.persist()
    // Build weak learner #0
    val wl0 = new CART(impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      1.0, 1.0).train(cart_ps)
    wk_learners += wl0.root
    // Update GBRT points
    gbrt_ps = gbrt_ps.map {
      p =>
        val f = CARTModel.predict(p.fs, wk_learners.last)
        GBRTPoint(p.label, f, p.fs)
    }.persist()
    // Create GBRT points for watchlist
    var gbrt_wl = watchlist.map {
      case (s, lps) =>
        val ps = lps.map {
          p =>
            val f = CARTModel.predict(p.fs, wk_learners.last)
            GBRTPoint(p.label, f, p.fs)
        }.persist()
        (s, ps)
    }
    // Compute RMSE
    val train_rmse = RMSE(gbrt_ps.map {
      p =>
        (p.label, p.f)
    })
    val watchlist_rmse = gbrt_wl.map {
      case (s, ps) =>
        (s, RMSE(ps.map {
          p =>
            (p.label, p.f)
        }))
    }
    // Show log
    val msg = new Msg()
    msg.append("RMSE(train)", train_rmse)
    watchlist_rmse.foreach {
      case (s, rmse) =>
        msg.append(s"RMSE($s)", rmse)
    }
    logInfo(s"GBRT Model round#1 training done,$msg")

    var iter = 1
    while (iter < num_round) {
      // Update CART points
      val cart_ps = gbrt_ps.map {
        p =>
          LabeledPoint(p.label - p.f, p.fs)
      }.persist()
      // Build weak learner #iter
      val wl = new CART(impurity,
        max_depth,
        max_bins,
        bin_samples,
        min_node_size,
        min_info_gain,
        1.0, 1.0).train(cart_ps)
      wk_learners += wl.root
      // Update GBRT points
      val pre_gbrt_ps = gbrt_ps
      gbrt_ps = GBRTPoint.update(pre_gbrt_ps, learn_rate, wk_learners.last).persist() // NOTICE!!! Must persist here!
      val pre_gbrt_wl = gbrt_wl
      gbrt_wl = pre_gbrt_wl.map {
        case (s, ps) =>
          val new_ps = GBRTPoint.update(ps, learn_rate, wk_learners.last).persist()
          (s, new_ps)
      }
      // Checkpoint every 20 iterations
      if (0 == iter % 20) {
        gbrt_ps.checkpoint()
        gbrt_wl.foreach(_._2.checkpoint())
      }
      // Force the materialization of this RDD
      gbrt_ps.count()
      pre_gbrt_ps.unpersist()
      gbrt_wl.foreach(_._2.count())
      pre_gbrt_wl.foreach(_._2.unpersist())
      // Compute RMSE
      val train_rmse = RMSE(gbrt_ps.map {
        p =>
          (p.label, p.f)
      })
      val watchlist_rmse = gbrt_wl.map {
        case (s, ps) =>
          (s, RMSE(ps.map {
            p =>
              (p.label, p.f)
          }))
      }
      // Show log
      val msg = new Msg()
      msg.append("RMSE(train)", train_rmse)
      watchlist_rmse.foreach {
        case (s, rmse) =>
          msg.append(s"RMSE($s)", rmse)
      }
      logInfo(s"GBRT Model round#${iter + 1} training done,$msg")
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