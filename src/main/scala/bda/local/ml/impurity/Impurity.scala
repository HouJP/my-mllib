package bda.local.ml.impurity

/**
 * Trait for calculate information gain.
 */
trait  Impurity {

  /**
   * Calculate information for regression
   *
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  def calculate(count: Double, sum: Double, sumSquares: Double): Double
}