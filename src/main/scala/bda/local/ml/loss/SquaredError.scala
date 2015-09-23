package bda.local.ml.loss

import bda.local.ml.model.Stat

class SquaredError {

  /** sum of SquaredError */
  var sumSE = 0.0
  /** num of SquaredError */
  var count = 0

  def :+(prediction: Double, label: Double): Unit = {
    sumSE += SquaredError.computeError(prediction, label)
    count += 1
  }

  def getRMSE: Double = {
    math.sqrt(sumSE / count)
  }

}

object SquaredError extends Loss {

  override def gradient(prediction: Double, label: Double): Double = {
    2.0 * (prediction - label)
  }

  override def computeError(prediction: Double, label: Double): Double = {
    val err = prediction - label
    err * err
  }

  override  def predict(stat: Stat): Double = {
    stat.sum / stat.count
  }
}