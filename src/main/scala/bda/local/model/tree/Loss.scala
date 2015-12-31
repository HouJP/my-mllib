package bda.local.model.tree

/**
 * Trait of loss functions for the gradient boosting algorithm.
 */
private[tree] trait LossCalculator extends Serializable {

  /**
   * Method to calculate the gradient.
   *
   * @param pred Prediction value.
   * @param label True label.
   * @return Gradient value.
   */
  def gradient(pred: Double, label: Double): Double

  /**
   * Method to calculate the squared error for the result of the model.
   *
   * @param pred Prediction value.
   * @param label True label.
   * @return The error of the predicted feature.
   */
  def computeError(pred: Double, label: Double): Double

  /**
   * Method to calculate the prediction of the tree node.
   *
   * @param sum The summation of labels in the tree node.
   * @param cnt The number of points in the tree node.
   * @return The prediction of the tree node.
   */
  def predict(sum: Double, cnt: Int): Double
}

/**
 * Class of squared error.
 */
private[tree] object SquaredErrorCalculator extends LossCalculator {

  /**
   * Method to calculate gradient for squared error.
   *
   * @param pred Prediction value.
   * @param label True label.
   * @return Gradient value.
   */
  override def gradient(pred: Double, label: Double): Double = {
    2.0 * (pred - label)
  }

  /**
   * Method to calculate squared error for predicted feature.
   *
   * @param pred Prediction value.
   * @param label True label.
   * @return The error of the predicted feature.
   */
  override def computeError(pred: Double, label: Double): Double = {
    val err = pred - label
    err * err
  }

  /**
   * Method to predict the tree node with squared error.
   *
   * @param sum The summation of labels in the tree node.
   * @param cnt The number of points in the tree node.
   * @return The prediction of the tree node.
   */
  override  def predict(sum: Double, cnt: Int): Double = {
    if (0 == cnt) {
      0.0
    } else {
      sum / cnt
    }
  }
}