package bda.local.ml.loss

import bda.local.ml.model.Stat

/**
 * Class for loss function of squared error.
 */
class SquaredErrorCounter extends LossCounter{

  /** sum of squared error */
  private var sum_e = 0.0
  /** num of instances */
  private var count = 0

  /**
   * Method to add squared error for a new instance.
   *
   * @param pre predicted feature
   * @param label true label
   */
  def :+=(pre: Double, label: Double): Unit = {
    sum_e += SquaredErrorCalculator.computeError(pre, label)
    count += 1
  }

  /**
   * Method to calculate RMSE on data
   *
   * @return RMSE value
   */
  def getMean: Double = {
    math.sqrt(sum_e / count)
  }
}

object SquaredErrorCalculator extends LossCalculator {

  /**
   * Method to calculate gradient for squared error
   *
   * @param pre predicted feature
   * @param label true label
   * @return gradient value
   */
  override def gradient(pre: Double, label: Double): Double = {
    2.0 * (pre - label)
  }

  /**
   * Method to calculate squared error for predicted feature
   *
   * @param pre predicted feature
   * @param label true label
   * @return the error of the predicted feature
   */
  override def computeError(pre: Double, label: Double): Double = {
    val err = pre - label
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