package bda.spark.ml.model

import bda.spark.ml.impurity.{ImpurityCalculator, VarianceCalculator}
import bda.spark.ml.loss.LossCalculator

class Stat(
    var count: Int,
    var sum: Double,
    var squared_sum: Double) extends Serializable {

  def update(label: Double): Unit = {
    count += 1
    sum += label
    squared_sum += label * label
  }

  def merge(other: Stat): Stat = {
    count += other.count
    sum += other.sum
    squared_sum += other.squared_sum

    this
  }

  def disunify(other: Stat): Stat = {
    count -= other.count
    sum -= other.sum
    squared_sum -= other.squared_sum

    this
  }

  def cal_prediction(loss_calculator: LossCalculator): Double = {
    loss_calculator.predict(sum, count)
  }

  def cal_impurity(impurity_calculator: ImpurityCalculator): Double = {
    impurity_calculator.calculate(count, sum, squared_sum)
  }

  def copy: Stat = {
    new Stat(count, sum, squared_sum)
  }

  override def toString: String = {
    s"count = $count, sum = $sum, squared_sum = $squared_sum}"
  }
}

object Stat {

  def empty: Stat = {
    new Stat(0, 0.0, 0.0)
  }

  def cal_weighted_impurity(stat_a: Stat, stat_b: Stat, impurity_calculator: ImpurityCalculator): Double = {
    val sum_count = stat_a.count + stat_b.count

    stat_a.cal_impurity(impurity_calculator) * stat_a.count / sum_count +
      stat_b.cal_impurity(impurity_calculator) * stat_b.count / sum_count
  }
}