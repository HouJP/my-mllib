package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint

private[cart] abstract class ImpurityAggregator(val off_set: Array[Int],
                                                val n_bins: Array[Int],
                                                val stats: Array[Double],
                                                val step: Int) extends Serializable {

  def update(point: CARTPoint, sub_fs: Array[Int])

  def calLeftInfo(id_f: Int, id_s: Int): (Double, Double, Double)

  def calRightInfo(id_f: Int, id_s: Int): (Double, Double, Double)

  def merge(other: ImpurityAggregator): ImpurityAggregator = {
    stats.indices.foreach {
      id =>
        stats(id) += other.stats(id)
    }
    this
  }

  def toPrefixSum(n_bins: Array[Int]): ImpurityAggregator = {
    n_bins.indices.foreach {
      id_f =>
        Range(off_set(id_f) + step, off_set(id_f) + n_bins(id_f) * step).foreach {
          id =>
            stats(id) += stats(id - step)
        }
    }
    this
  }

  override def toString = {
    s"off_set(${off_set.mkString(",")}), stats(${stats.mkString(",")})"
  }
}