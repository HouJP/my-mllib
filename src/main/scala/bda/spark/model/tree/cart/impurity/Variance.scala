package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint
import org.apache.spark.rdd.RDD

private[cart] object Variance extends Impurity {

  def stat(cart_ps: RDD[CARTPoint]): ImpurityStatus = {
    val n = cart_ps.map(_.weight).sum.toInt
    val sum = cart_ps.map(p => p.label * p.weight).sum
    val squared_sum = cart_ps.map(p => p.label * p.label * p.weight).sum
    val stt = Array(sum, squared_sum)
    new VarianceStatus(n, stt)
  }

  def calculate(stat: ImpurityStatus): Double = {
    if (0 == stat.num) {
      0
    } else {
      (stat.stt(1) - math.pow(stat.stt(0), 2) / stat.num) / stat.num
    }
  }

  def predict(stat: ImpurityStatus): Double = {
    if (0 == stat.num) {
      0.0
    } else {
      stat.stt(0) / stat.num
    }
  }

  def agg(stat: ImpurityStatus,
          n_bins: Array[Int]): ImpurityAggregator = {

    val step = stat.stt.length
    val off_fs = n_bins.scanLeft(0)((sum, n_bin) => sum + n_bin * step)
    val tol_bins = n_bins.sum
    val stt = new Array[Double](tol_bins * step)

    new VarianceAggregator(off_fs, n_bins, stt, 3)
  }
}