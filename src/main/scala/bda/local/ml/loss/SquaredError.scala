package bda.local.ml.loss

import bda.local.ml.model.Stat

/**
 * Class for loss function of squared error.
 */
class SquaredError {

  /** sum of squared error */
  var sumSE = 0.0
  /** num of instances */
  var count = 0

  /**
   * Method to add squared error for a new instance.
   *
   * @param prediction predicted feature
   * @param label true label
   */
  def :+(prediction: Double, label: Double): Unit = {
    sumSE += SquaredError.computeError(prediction, label)
    count += 1
  }

  /**
   * Method to calculate RMSE on data
   *
   * @return RMSE value
   */
  def getRMSE: Double = {
    math.sqrt(sumSE / count)
  }
}

object SquaredError extends Loss {

  /**
   * Method to calculate gradient for squared error
   *
   * @param prediction predicted feature
   * @param label true label
   * @return gradient value
   */
  override def gradient(prediction: Double, label: Double): Double = {
    2.0 * (prediction - label)
  }

  /**
   * Method to calculate squared error for predicted feature
   *
   * @param prediction predicted feature
   * @param label true label
   * @return the error of the predicted feature
   */
  override def computeError(prediction: Double, label: Double): Double = {
    val err = prediction - label
    err * err
  }

  /**
   * Mathod to predict feature with squared error
   *
   * @param stat the stat of a leaf in a decision tree [[bda.local.ml.model.Stat]]
   * @return the predicted feature
   */
  override  def predict(stat: Stat): Double = {
    stat.sum / stat.count
  }
}