package bda.spark.model.tree

/**
 * Trait to calculate information gain.
 */
private[tree] trait ImpurityCalculator extends Serializable {

  /**
   * Calculate information for regression
   *
   * @param count Number of instances.
   * @param sum Summation of labels.
   * @param sum_squares Summation of squares of labels.
   * @return Information value, or 0 if count = 0
   */
  def calculate(count: Double, sum: Double, sum_squares: Double): Double

}

/**
 * Class for calculating variance during regression
 */
private[tree] object VarianceCalculator extends  ImpurityCalculator {

  /**
   * Variance calculation.
   *
   * @param count Number of instances.
   * @param sum Summation of labels.
   * @param sum_squares Summation of squares of the labels.
   * @return Information value, or 0 if count = 0.
   */
  override def calculate(count: Double, sum: Double, sum_squares: Double): Double = {
    if (0 == count) {
      return 0
    }
    val squared_loss = sum_squares - (sum * sum) / count
    squared_loss / count
  }
}