package bda.local.ml

import bda.local.ml.para.{GBoostPara, DTreePara}
import bda.local.ml.model.{GBoostModel, DTreeModel, LabeledPoint}
import bda.local.ml.util.Log

class GBoost(val gb_para: GBoostPara) {

  def fit(input: Array[LabeledPoint]): GBoostModel = {
    val num_iter = gb_para.num_iter
    val learn_rate = gb_para.learn_rate
    val dt_para = gb_para.dt_para
    val loss_calculator = gb_para.loss_calculator
    val min_step = gb_para.min_step
    val size = input.length
    var data = input

    val wk_learners = new Array[DTreeModel](num_iter)

    var cost_time = 0.0
    var cost_count = 0

    val begin_t = System.nanoTime()

    // building weak learner #0
    val wl0 = new DTree(dt_para).fit(data)
    wk_learners(0) = wl0

    // compute prediction and error
    val pred_err = wl0.computePredictAndError(input, learn_rate)
    var pred = pred_err._1
    var err = pred_err._2

    // compute mean error
    var min_err = err
    var best_iter = 1

    val end_t = System.nanoTime()
    val bias_t = (end_t - begin_t) / 1e6
    cost_time += bias_t
    cost_count += 1
    Log.log("INFO", s"fitting: iter = 0, error = ${min_err}, cost_time = $bias_t")

    data = pred.zip(input).map { case (predict, lp) =>
      LabeledPoint(-1.0 * loss_calculator.gradient(predict, lp.label), lp.features)
    }

    var iter = 1
    while (iter < num_iter) {
      val begin_t = System.nanoTime()

      // building weak leaner #iter
      val wl = new DTree(dt_para).fit(data)
      wk_learners(iter) = wl

      // compute prediction and error
      val predError = wl.updatePredictAndError(input, pred, learn_rate)
      pred = predError._1
      err = predError._2

      // compute mean error
      val current_err = err

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

      // update data with pseudo-residuals
      data = pred.zip(input).map { case (predict, lp) =>
        LabeledPoint(-1.0 * loss_calculator.gradient(predict, lp.label), lp.features)
      }

      iter += 1
    }

    Log.log("INFO", s"GBoost model training done, average cost time of each iteration: ${cost_time / cost_count}")
    new GBoostModel(wk_learners.slice(0, best_iter), gb_para)
  }
}