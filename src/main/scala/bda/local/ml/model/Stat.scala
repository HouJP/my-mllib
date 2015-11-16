package bda.local.ml.model

import bda.local.ml.impurity.{ImpurityCalculator, VarianceCalculator}

/**
 * Class of status of the node in a tree.
 *
 * @param impurity_calculator impurity calculator [[bda.local.ml.impurity.ImpurityCalculator]]
 * @param count number of instances the node has
 * @param sum summation of labels of instances the node has
 * @param squared_sum summation of squares of labels of instances the node has
 * @param left_index leftmost id of instances the node has
 * @param right_index next id of rightmost instances the node has
 */
class Stat(
    var impurity_calculator: ImpurityCalculator,
    var count: Int,
    var sum: Double,
    var squared_sum: Double,
    var left_index: Int,
    var right_index: Int) {

  /** information value of the node */
  var impurity = impurity_calculator.calculate(count, sum, squared_sum)

  override def toString: String = {
    s"count = $count, sum = $sum, sumSquares = $squared_sum, " +
      s"leftIndex = $left_index, rightIndex = $right_index, " +
      s"impurity = $impurity"
  }

  /**
   * Method to copy another stat.
   *
   * @param stat stat of another node
   */
  def copy(stat: Stat): Unit = {
    this.impurity_calculator = stat.impurity_calculator
    this.count = stat.count
    this.sum = stat.sum
    this.squared_sum = stat.squared_sum
    this.left_index = stat.left_index
    this.right_index = stat.right_index
    this.impurity = stat.impurity
  }

  /**
   * Method to udpate stat of the node with variations.
   *
   * @param count_bias count variation of the node
   * @param sum_bias sum variation of the node
   * @param sum_squares_bias sumSquares variation of the node
   * @param leftIndexBias left index variation of the node
   * @param rightIndexBias right index variation of the node
   */
  def update(
      count_bias: Int,
      sum_bias: Double,
      sum_squares_bias: Double,
      leftIndexBias: Int,
      rightIndexBias: Int): Unit = {

    count += count_bias
    sum += sum_bias
    squared_sum += sum_squares_bias
    left_index += leftIndexBias
    right_index += rightIndexBias
    impurity = impurity_calculator.calculate(count, sum, squared_sum)
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
