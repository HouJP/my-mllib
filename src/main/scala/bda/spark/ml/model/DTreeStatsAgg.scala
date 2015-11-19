package bda.spark.ml.model

import bda.spark.ml.impurity.ImpurityCalculator
import bda.spark.ml.loss.LossCalculator
import bda.spark.ml.para.DTreePara

class DTreeStatsAgg(metadata: DTreeMetaData) extends Serializable {

  val step_stat = 3
  val num_stats = metadata.num_bins.sum
  val num_fs = metadata.num_features
  val len_stats = step_stat * num_stats
  val stats = new Array[Double](len_stats)

  val off_fs = {
    metadata.num_bins.scanLeft(0)((sum, num_bins) => sum + num_bins * step_stat)
  }

  def update(label: Double, index_f: Int, binned_f: Int): Unit = {
    val offset = off_fs(index_f) + 3 * binned_f
    stats(offset + 0) += 1
    stats(offset + 1) += label
    stats(offset + 2) += label * label
  }

  def update(p: DTreePoint): Unit = {
    var index_f = 0
    while (index_f < num_fs) {
      update(p.label, index_f, p.binned_features(index_f))
      index_f += 1
    }
  }

  def merge(other: DTreeStatsAgg): DTreeStatsAgg = {
    var index = 0
    while (index < len_stats) {
      stats(index) += other.stats(index)
      index += 1
    }

    this
  }

  def toPrefixSum(): DTreeStatsAgg = {
    Range(0, num_fs).foreach { index_f =>
      val num_bin = metadata.num_bins(index_f)

      var index_b = 1
      while (index_b < num_bin) {
        val offset = off_fs(index_f) + 3 * index_b
        stats(offset + 0) += stats(offset + 0 - 3)
        stats(offset + 1) += stats(offset + 1 - 3)
        stats(offset + 2) += stats(offset + 2 - 3)
        index_b += 1
      }
    }
    this
  }

  def calLeftInfo(
      index_f: Int,
      index_b: Int,
      impurity_calculator: ImpurityCalculator,
      loss_calculator: LossCalculator): (Double, Double, Int) = {

    val off = off_fs(index_f) + 3 * index_b

    val count = stats(off).toInt
    val sum = stats(off + 1)
    val squared_sum = stats(off + 2)

    val impurity = impurity_calculator.calculate(count, sum, squared_sum)
    val predict = loss_calculator.predict(sum, count)

    (impurity, predict, count)
  }

  def calRightInfo(
      index_f: Int,
      index_b: Int,
      impurity_calculator: ImpurityCalculator,
      loss_calculator: LossCalculator): (Double, Double, Int) = {

    val off = off_fs(index_f) + 3 * index_b
    val last_off = off_fs(index_f) + 3 * (metadata.num_bins(index_f) - 1)

    val count = (stats(last_off) - stats(off)).toInt
    val sum = stats(last_off + 1) - stats(off + 1)
    val squared_sum = stats(last_off + 2) - stats(off + 2)

    val impurity = impurity_calculator.calculate(count, sum, squared_sum)
    val predict = loss_calculator.predict(sum, count)

    (impurity, predict, count)
  }
}