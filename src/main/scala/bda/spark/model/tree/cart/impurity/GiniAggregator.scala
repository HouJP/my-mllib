package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint

private[cart] class GiniAggregator(off_fs: Array[Int],
                                   n_bins: Array[Int],
                                   stats: Array[Double],
                                   step: Int,
                                   map_label: Map[Double, Int]) extends ImpurityAggregator(off_fs, n_bins, stats, step) {

  def update(label: Double, weight: Int, id_f: Int, binned_f: Int): Unit = {
    val off_b = off_fs(id_f) + step * binned_f
    val off_l = off_b + map_label(label) + 1
    stats(off_b) += weight
    stats(off_l) += weight
    if (off_b == 4 || off_l == 4)
    println(s"lable($label),weight($weight),id_f($id_f),binned_f($binned_f),off_b($off_b),off_l($off_l)")
  }

  def update(point: CARTPoint, sub_fs: Array[Int]) = {
    sub_fs.foreach {
      id_f =>
        update(point.label, point.weight, id_f, point.binned_fs(id_f))
    }
  }

  def calLeftInfo(id_f: Int, id_s: Int): (Double, Double, Double) = {
    val off = off_fs(id_f) + step * id_s

    val sub_stat = stats.slice(off, off + step)
    val total = sub_stat(0)
    val impurity = Gini.calculate(sub_stat)
    val predict = Gini.predict(sub_stat, map_label)

    (impurity, predict, total)
  }

  def calRightInfo(id_f: Int, id_s: Int): (Double, Double, Double) = {
    val off = off_fs(id_f) + step * id_s
    val off_last = off_fs(id_f) + step * (n_bins(id_f) - 1)

    val sub_stat = Range(0, step).map {
      id =>
        stats(off_last + id) - stats(off + id)
    }.toArray
    val total = sub_stat(0)
    val impurity = Gini.calculate(sub_stat)
    val predict = Gini.predict(sub_stat, map_label)

    (impurity, predict, total)
  }
}