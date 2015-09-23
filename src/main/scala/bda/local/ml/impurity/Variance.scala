package bda.local.ml.impurity

import bda.local.ml.model.Stat

object Variance extends  Impurity {

  override def calculate(count: Double, sum: Double, sumSquares: Double): Double = {
    if (0 == count) {
      return 0
    }
    val squaredLoss = sumSquares - (sum * sum) / count
    squaredLoss / count
  }
}