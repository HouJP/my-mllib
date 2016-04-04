package bda.spark.model.tree.gbdt

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import org.apache.spark.rdd.RDD

private[gbdt] case class GBDTPoint(label: Int,
                                   fv: Array[Double],
                                   fs: SparseVector[Double]) extends Serializable {

  override def toString = {
    s"label($label), fv(${fv.mkString(",")}), fs($fs)"
  }
}

private[gbdt] object GBDTPoint {

  def toGBDTPoint(lps: RDD[LabeledPoint], n_label: Int): RDD[GBDTPoint] = {
    lps.map {
      lp =>
        GBDTPoint(lp.label.toInt, Array.fill[Double](n_label)(0.0), lp.fs)
    }
  }
}