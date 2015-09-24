package bda.local.ml.impurity

/**
 * Class for calculating variance during regression
 */
object Variance extends  Impurity {

  /**
   * variance calculation
   *
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double = {
    if (0 == count) {
      return 0
    }
    val squaredLoss = sumSquares - (sum * sum) / count
    squaredLoss / count
  }
}