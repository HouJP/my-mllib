package bda.local.ml.loss

import bda.local.ml.model.Stat

abstract class LossCounter {

  /**
   * Method to add error for a new instance.
   *
   * @param pre predicted feature
   * @param label true label
   */
  def :+=(pre: Double, label: Double): Unit

  /**
   * Method to calculate mean error on data
   *
   * @return RMSE value
   */
  def getMean: Double
}

/**
 * Trait for loss functions for the gradient boosting algorithm
 */
trait LossCalculator extends Serializable {

  /**
   * Method to calculate the gradient for the gradient boosting algorithm
   *
   * @param pre predicted feature
   * @param label true label
   * @return gradient value
   */
  def gradient(pre: Double, label: Double): Double

  /**
   * Method to calculate the error for the result of the model
   *
   * @param pre predicted feature
   * @param label true label
   * @return the error of the predicted feature
   */
  def computeError(pre: Double, label: Double): Double

  /**
   * Method to calculate the predicted feature
   * Different predicted feature can be get through differenct loss functions
   *
   * @param stat the stat of a leaf in a decision tree [[bda.local.ml.model.Stat]]
   * @return the predicted feature
   */
  def predict(stat: Stat): Double
}