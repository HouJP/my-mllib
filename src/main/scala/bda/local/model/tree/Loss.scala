package bda.local.model.tree

private[tree] abstract class LossCounter {

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
private[tree] trait LossCalculator extends Serializable {

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
   * @param stat the stat of a leaf in a decision tree [[bda.local.model.tree.DecisionTreeStat]]
   * @return the predicted feature
   */
  def predict(stat: DecisionTreeStat): Double
}


/**
 * Class for loss function of squared error.
 */
private[tree] class SquaredErrorCounter extends LossCounter{

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

private[tree] object SquaredErrorCalculator extends LossCalculator {

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
   * @param stat the stat of a leaf in a decision tree [[bda.local.model.tree.DecisionTreeStat]]
   * @return the predicted feature
   */
  override  def predict(stat: DecisionTreeStat): Double = {
    stat.sum / stat.count
  }
}