package bda.local.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.loss.{SquaredError, Loss}
import bda.local.ml.strategy.GBoostStrategy
import bda.local.ml.util.Log

/**
 * Class of GBoost model which stored GBoost model structure and strategy.
 */
class GBoostModel(
    weakLearners: Array[DTreeModel],
    gBoostStrategy: GBoostStrategy) {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features feature vector of a single data point
   * @return predicted value from the trained model
   */
  def predict(features: SparseVector[Double]): Double = {
    val preds = weakLearners.map(_.predict(features))
    preds.map(_ * gBoostStrategy.learningRate).sum
  }

  /**
   * Predict values for the given data using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input Array of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points
   * @return Array stored prediction
   */
  def predict(input: Array[LabeledPoint]): Array[Double] = {
    val se = new SquaredError

    val output = input.map { p =>
      val pre = predict(p.features)
      se :+= (pre, p.label)
      pre
    }

    Log.log("INFO", s"predict done, with RMSE = ${se.getRMSE}")

    output
  }
}

object GBoostModel {
  def computeInitialPredictAndError(
      input: Array[LabeledPoint],
      wl0: DTreeModel,
      weight: Double,
      loss: Loss): (Array[Double], Array[Double]) = {
    val predError = input.map { lp =>
      val predict = wl0.predict(lp.features) * weight
      val error = loss.computeError(predict, lp.label)
      (predict, error)
    }
    val pred = predError.map(_._1)
    val error = predError.map(_._2)
    (pred, error)
  }

  def updatePredictAndError(
      input: Array[LabeledPoint],
      prePred: Array[Double],
      wl: DTreeModel,
      weight: Double,
      loss: Loss): (Array[Double], Array[Double]) = {
    val predError = input.zip(prePred).map { case (lp, prePredict) =>
        val predict = prePredict + wl.predict(lp.features) * weight
        val error = loss.computeError(predict, lp.label)
      (predict, error)
    }

    val pred = predError.map(_._1)
    val error = predError.map(_._2)

    (pred, error)
  }
}