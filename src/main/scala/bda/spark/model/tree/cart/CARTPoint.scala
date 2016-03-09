package bda.spark.model.tree.cart

import bda.common.obj.LabeledPoint
import bda.common.util.PoissonDistribution
import org.apache.spark.rdd.RDD

private[cart] case class CARTPoint(label: Double,
                              weight: Int,
                              binned_fs: Array[Int]) extends Serializable {

  override def toString = {
    s"label($label), weight($weight), binned_fs(${binned_fs.mkString(",")})"
  }
}

private[cart] object CARTPoint {

  def toCARTPoint(lps: RDD[LabeledPoint],
                  splits: Array[Array[CARTSplit]],
                  n_fs: Int,
                  row_rate: Double): RDD[CARTPoint] = {

    lps.mapPartitions {
      iter =>
        // Set points' weight using poisson distribution
        val ps = new PoissonDistribution(row_rate)

        iter.map {
          lp =>
            CARTPoint.toCARTPoint(lp, splits, n_fs, ps)
        }
    }
  }

  def toCARTPoint(lp: LabeledPoint,
                  splits: Array[Array[CARTSplit]],
                  n_fs: Int,
                  ps: PoissonDistribution): CARTPoint = {

    val binned_fs = new Array[Int](n_fs)
    Range(0, n_fs).foreach {
      id_f =>
        val binned_f = binarySearchForBin(lp.fs(id_f), splits, id_f)
        binned_fs(id_f) = binned_f
    }

    val weight = {
      if (ps.getMean < 1.0) {
        ps.sample()
      } else {
        1
      }
    }

    new CARTPoint(lp.label, weight, binned_fs)
  }

  def binarySearchForBin(v: Double,
                         splits: Array[Array[CARTSplit]],
                         id_f: Int): Int = {

    var l = 0
    var r = splits(id_f).length - 1

    while (l <= r) {
      val mid = (l + r) / 2
      if (splits(id_f)(mid).threshold <= v) {
        l = mid + 1
      } else {
        r = mid - 1
      }
    }
    l
  }
}