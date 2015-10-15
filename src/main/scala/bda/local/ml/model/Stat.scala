package bda.local.ml.model

import bda.local.ml.impurity.{ImpurityCalculator, VarianceCalculator}

/**
 * Class of status of the node in a tree.
 *
 * @param impurityCalculator impurity calculator [[bda.local.ml.impurity.ImpurityCalculator]]
 * @param count number of instances the node has
 * @param sum summation of labels of instances the node has
 * @param sumSquares summation of squares of labels of instances the node has
 * @param leftIndex leftmost id of instances the node has
 * @param rightIndex next id of rightmost instances the node has
 */
class Stat(
    var impurityCalculator: ImpurityCalculator,
    var count: Int,
    var sum: Double,
    var sumSquares: Double,
    var leftIndex: Int,
    var rightIndex: Int) {

  /** information value of the node */
  var impurity = impurityCalculator.calculate(count, sum, sumSquares)

  override def toString: String = {
    s"count = $count, sum = $sum, sumSquares = $sumSquares, " +
      s"leftIndex = $leftIndex, rightIndex = $rightIndex, " +
      s"impurity = $impurity"
  }

  /**
   * Method to copy another stat.
   *
   * @param stat stat of another node
   */
  def copy(stat: Stat): Unit = {
    this.impurityCalculator = stat.impurityCalculator
    this.count = stat.count
    this.sum = stat.sum
    this.sumSquares = stat.sumSquares
    this.leftIndex = stat.leftIndex
    this.rightIndex = stat.rightIndex
    this.impurity = stat.impurity
  }

  /**
   * Method to udpate stat of the node with variations.
   *
   * @param countBias count variation of the node
   * @param sumBias sum variation of the node
   * @param sumSquaresBias sumSquares variation of the node
   * @param leftIndexBias left index variation of the node
   * @param rightIndexBias right index variation of the node
   */
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

  /**
   * Method to add a instance which is next to the leftmost instance of the node.
   *
   * @param value the true label of the instance which will be added
   */
  def +=:(value: Double): Unit = {
    update(1, value, value * value, -1, 0)
  }

  /**
   * Method to add a instance which is next to the rightmost instance of the node.
   *
   * @param value the true label of the instance which will be added
   */
  def :+=(value: Double): Unit = {
    update(1, value, value * value, 0, 1)
  }

  /**
   * Method to subtract a instance which is next to the leftmost instance of the node.
   *
   * @param value the true label of the instance which will be subtracted
   */
  def -=:(value: Double): Unit = {
    update(-1, -1 * value, -1 * value * value, 1, 0)
  }

  /**
   * Method to subtract a instance which is next to the rightmost instance of the node.
   *
   * @param value the true label of the instance which will be subtracted
   */
  def :-=(value: Double): Unit = {
    update(-1, -1 * value, -1 * value * value, 0, -1)
  }
}

object Stat {

  /**
   * Construct a [[Stat]] instance with original value.
   *
   * @return a [[Stat]] instance
   */
  def empty = {
    new Stat(VarianceCalculator, 0, 0, 0, 0, 0)
  }
}
