package bda.local.ml.impurity

trait  Impurity {

  def calculate(count: Double, sum: Double, sumSquares: Double): Double
}