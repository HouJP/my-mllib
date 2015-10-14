package bda.spark.ml

import bda.local.ml.DTree
import bda.spark.ml.model.GBoostModel
import bda.spark.ml.para.GBoostPara
import bda.local.ml.model.{DTreeModel, LabeledPoint}
import bda.local.ml.util.Log
import org.apache.spark.rdd.RDD
import scala.util.Random

/**
 * A class which implement gradient boosting algorithm using [[bda.local.ml.DTree]] as weak learners.
 *
 * @param gb_para the configuration parameters for the Gradient Boosting algorithm
 */
class GBoost(val gb_para: GBoostPara) {

  /**
   * Method to train a gradient boosting model over a training data which represented as an array of [[bda.local.ml.model.LabeledPoint]].
   *
   * @param input traning data: RDD of [[bda.local.ml.model.LabeledPoint]]
   * @return a [[bda.local.ml.model.GBoostModel]] instance which can be used to predict.
   */
  def fit(input: RDD[LabeledPoint]): GBoostModel = {
    val num_iter = gb_para.num_iter
    val data_sample_rate = gb_para.data_sample_rate
    val max_data_sample = gb_para.max_data_sample
    val learn_rate = gb_para.learn_rate
    val dt_para = gb_para.dt_para
    val loss_calculator = gb_para.loss_calculator
    val min_step = gb_para.min_step
    val size = input.count()
    var sample_size = math.min(size * data_sample_rate, max_data_sample).toInt

    val wk_learners = new Array[DTreeModel](num_iter)

    // persist input RDD for reusing
    input.persist()
    //input.foreachPartition(_ => Unit)

    var cost_time = 0.0
    var cost_count = 0

    val begin_t = System.nanoTime()

    // sample input data
    var data = input.takeSample(withReplacement = false, sample_size, new Random().nextLong())

    // build weak learner 0
    val wl0 = new DTree(dt_para).fit(data)
    wk_learners(0) = wl0

    // compute prediction and error
    var pred_err = GBoostModel.computePredictAndError(input, learn_rate, wl0, loss_calculator).persist()
    //pred_err.foreachPartition(_ => Unit)

    // compute mean error
    var min_err = pred_err.values.mean()
    var best_iter = 1

    val end_t = System.nanoTime()
    val bias_t = (end_t - begin_t) / 1e6
    cost_time += bias_t
    cost_count += 1
    Log.log("INFO", s"fitting: iter = 0, error = $min_err, cost_time = $bias_t")

    var iter = 1
    while (iter < num_iter) {
      val begin_t = System.nanoTime()

      // sample input data
      data = pred_err.zip(input).map { case ((pred, _), lp) =>
        LabeledPoint(-1.0 * loss_calculator.gradient(pred, lp.label), lp.features)
      }.takeSample(withReplacement = false, sample_size, new Random().nextLong())

      // building weak leaner #iter
      val wl = new DTree(dt_para).fit(data)
      wk_learners(iter) = wl

      // compute prediction and error
      val pre_pred_err = pred_err
      pred_err = GBoostModel.updatePredictAndError(input, pre_pred_err, learn_rate, wl, loss_calculator).persist()
      //pred_err.foreachPartition(_ => Unit)
      pre_pred_err.unpersist()

      // compute mean error
      val current_err = pred_err.values.mean()

      val end_t = System.nanoTime()
      val bias_t = (end_t - begin_t) / 1e6
      cost_time += bias_t
      cost_count += 1
      Log.log("INFO", s"fitting: iter = $iter, error = $current_err, cost_time = $bias_t")

      if (min_err - current_err < min_step) {
        Log.log("INFO", s"GBoost model training done, average cost time of each iteration: ${cost_time / cost_count}")
        return new GBoostModel(wk_learners.slice(0, best_iter), gb_para)
      } else if (current_err < min_err) {
        min_err = current_err
        best_iter = iter + 1
      }

      iter += 1
    }

    Log.log("INFO", s"GBoost model training done, average cost time of each iteration: ${cost_time / cost_count}")

    // unpersist input RDD
    input.unpersist()

    new GBoostModel(wk_learners.slice(0, best_iter), gb_para)
  }
}

object GBoost {
}