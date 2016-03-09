package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint
import org.apache.spark.rdd.RDD

private[cart] object Gini extends Impurity {

  def stat(cart_ps: RDD[CARTPoint]): ImpurityStatus = {
    val num = cart_ps.map(_.weight).sum.toInt
    val cnt = cart_ps.map(p => (p.label, p.weight.toDouble)).reduceByKey(_+_).collectAsMap()
    val stt = cnt.values.toArray
    val map_label = cnt.keys.zipWithIndex.toMap
    val num_label = stt.length
    new GiniStatus(num, stt, map_label, num_label)
  }

  def calculate(stat: ImpurityStatus): Double = {
    if (0 == stat.num) {
      1.0
    } else {
      val sum_p = stat.stt.map {
        num =>
          math.pow(num / stat.num, 2.0)
      }.sum
      1.0 - sum_p
    }
  }

  def calculate(stat: Array[Double]): Double = {
    require(stat.size > 2,
      "require length(status array) > 2 when calculate gini impurity.")

    val tol = stat(0)

    if (0 == tol) {
      1.0
    } else {
      1.0 - Range(1, stat.length).map {
        id =>
          math.pow(stat(id) / tol, 2)
      }.sum
    }
  }

  def predict(stat: ImpurityStatus): Double = {
    val real_stat = stat.asInstanceOf[GiniStatus]
    real_stat.map_label.mapValues {
      id =>
        real_stat.stt(id)
    }.maxBy(_._2)._1
  }

  def predict(stat: Array[Double], map_label: Map[Double, Int]): Double = {
    map_label.map {
      case (label, pos) =>
        (label, stat(pos + 1))
    }.maxBy(_._2)._1
  }

  def agg(stat: ImpurityStatus,
          num_bins: Array[Int]): ImpurityAggregator = {

    val gini_stat = stat.asInstanceOf[GiniStatus]
    val step = stat.stt.length + 1
    val off_fs = num_bins.scanLeft(0)((sum, n_bin) => sum + n_bin * step)
    val tol_bins = num_bins.sum
    val stt = Array.fill[Double](tol_bins * step)(0.0)

    new GiniAggregator(off_fs, num_bins, stt, step, gini_stat.map_label)
  }
}