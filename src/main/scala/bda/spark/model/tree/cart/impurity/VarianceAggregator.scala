package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint

private[cart] class VarianceAggregator(off_fs: Array[Int],
                                       n_bins: Array[Int],
                                       stats: Array[Double],
                                       step: Int) extends ImpurityAggregator(off_fs, n_bins, stats, step) {

  def update(point: CARTPoint, sub_fs: Array[Int]) = {

  }

  def calLeftInfo(id_f: Int, id_s: Int): (Double, Double, Double) = {
    (0.0, 0.0, 0.0)
  }

  def calRightInfo(id_f: Int, id_s: Int): (Double, Double, Double) = {
    (0.0, 0.0, 0.0)
  }
}