package bda.local.ml.loss

import bda.local.ml.model.Stat

/**
 * Trait for loss functions for the gradient boosting algorithm
 */
trait Loss {

  /**
   * Method to calculate the gradient for the gradient boosting algorithm
   *
   * @param prediction predicted feature
   * @param label true label
   * @return gradient value
   */
  def gradient(prediction: Double, label: Double): Double

  /**
   * Method to calculate the error for the result of the model
   *
   * @param prediction predicted feature
   * @param label true label
   * @return the error of the predicted feature
   */
  def computeError(prediction: Double, label: Double): Double

  /**
   * Method to calculate the predicted feature
   * Different predicted feature can be get through differenct loss functions
   *
   * @param stat the stat of a leaf in a decision tree [[bda.local.ml.model.Stat]]
   * @return the predicted feature
   */
  def predict(stat: Stat): Double
}