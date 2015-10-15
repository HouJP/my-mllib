package bda.local.ml.impurity

/**
 * Class for calculating variance during regression
 */
object VarianceCalculator extends  ImpurityCalculator {

  /**
   * variance calculation
   *
   * @param count number of instances
   * @param sum sum of labels
   * @param sum_squares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  override def calculate(count: Double, sum: Double, sum_squares: Double): Double = {
    if (0 == count) {
      return 0
    }
    val squared_loss = sum_squares - (sum * sum) / count
    squared_loss / count
  }
}