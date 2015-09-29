package bda.local.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.loss.SquaredErrorCounter
import bda.local.ml.para.GBoostPara
import bda.local.ml.para.Loss._
import bda.local.ml.util.Log

/**
 * Class of GBoost model which stored GBoost model structure and strategy.
 *
 * @param wk_learners weak learners which formed gradient boosting model
 * @param gb_para the configuration parameters for gradient boosting algorithm
 */
class GBoostModel(
    wk_learners: Array[DTreeModel],
    gb_para: GBoostPara) {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point
   * @return predicted value from the trained model
   */
  def predict(fs: SparseVector[Double]): Double = {
    val preds = wk_learners.map(_.predict(fs))
    preds.map(_ * gb_para.learn_rate).sum
  }

  /**
   * Predict values for the given data using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input Array of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points
   * @return Array stored prediction
   */
  def predict(input: Array[LabeledPoint]): Array[Double] = {
    val err_counter = gb_para.loss match {
      case SquaredError => new SquaredErrorCounter()
      case _ => throw new IllegalArgumentException(s"Did not recognize loss type: ${gb_para.loss}")
    }

    val output = input.map { p =>
      val pre = predict(p.features)
      err_counter :+= (pre, p.label)
      pre
    }

    Log.log("INFO", s"predict done, with RMSE = ${err_counter.getMean}")

    output
  }
}

object GBoostModel {

}