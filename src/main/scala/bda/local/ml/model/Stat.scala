package bda.local.ml.model

import bda.local.ml.impurity.{Variance, Impurity}

class Stat(
    var impurityCalculator: Impurity,
    var count: Int,
    var sum: Double,
    var sumSquares: Double,
    var leftIndex: Int,
    var rightIndex: Int) {

  var impurity = impurityCalculator.calculate(count, sum, sumSquares)

  override def toString: String = {
    s"count = $count, sum = $sum, sumSquares = $sumSquares, " +
      s"leftIndex = $leftIndex, rightIndex = $rightIndex, " +
      s"impurity = $impurity"
  }

  def copy(stat: Stat): Unit = {
    this.impurityCalculator = stat.impurityCalculator
    this.count = stat.count
    this.sum = stat.sum
    this.sumSquares = stat.sumSquares
    this.leftIndex = stat.leftIndex
    this.rightIndex = stat.rightIndex
    this.impurity = stat.impurity
  }

  def update(
      countBias: Int,
      sumBias: Double,
      sumSquaresBias: Double,
      leftIndexBias: Int,
      rightIndexBias: Int): Unit = {

    count += countBias
    sum += sumBias
    sumSquares += sumSquaresBias
    leftIndex += leftIndexBias
    rightIndex += rightIndexBias
    impurity = impurityCalculator.calculate(count, sum, sumSquares)
  }

  def +:(value: Double): Unit = {
    update(1, value, value * value, -1, 0)
  }

  def :+(value: Double): Unit = {
    update(1, value, value * value, 0, 1)
  }

  def -:(value: Double): Unit = {
    update(-1, -1 * value, -1 * value * value, 1, 0)
  }

  def :-(value: Double): Unit = {
    update(-1, -1 * value, -1 * value * value, 0, -1)
  }
}

object Stat {
  def empty = {
    new Stat(Variance, 0, 0, 0, 0, 0)
  }
}
