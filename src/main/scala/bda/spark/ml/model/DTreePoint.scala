package bda.spark.ml.model

import bda.local.ml.model.LabeledPoint
import bda.common.linalg.immutable.SparseVector
import org.apache.spark.rdd.RDD

class DTreePoint(val label: Double, val features: SparseVector[Double], val binned_features: Array[Int]) {

  override def toString: String = {
    var str_binned_fs = new String()
    for (index_f <- 0 until features.size) {
      str_binned_fs += s" ${index_f + 1}:${features(index_f)}, ${binned_features(index_f)}"
    }
    s"${label}" + str_binned_fs
  }
}

object DTreePoint {

  def convertToTreeRDD(input: RDD[LabeledPoint], bins: Array[Array[Bin]], metadata: DTreeMetaData): RDD[DTreePoint] = {
    input.map{ case lp =>
      DTreePoint.convertToTreePoint(lp, bins, metadata)
    }
  }

  def convertToTreePoint(
      lp: LabeledPoint,
      bins: Array[Array[Bin]],
      metadata: DTreeMetaData): DTreePoint = {

    val num_features = metadata.num_features
    val binned_fs = new Array[Int](num_features)
    for (index_f <- 0 until num_features) {
      val binned_f = binarySearchForBin(lp.features(index_f), index_f, bins, metadata)

      // check the binned feature
      require(-1 != binned_f, s"LabeledPoint with label = ${lp.label} couldn't find correct bin" +
        s" with feature-index = $index_f, feature-value = ${lp.features(index_f)}")

      // convert the feature to binned feature
      binned_fs(index_f) = binned_f
    }

    new DTreePoint(lp.label, lp.features, binned_fs)
  }

  def binarySearchForBin(
      value: Double,
      index_feature: Int,
      bins: Array[Array[Bin]],
      metadata: DTreeMetaData): Int = {

    var index_l = 0
    var index_r = metadata.num_bins(index_feature) - 1
    while (index_l <= index_r) {
      val index_m = (index_l + index_r) / 2
      if (bins(index_feature)(index_m).low_split.threshold >= value) {
        index_r = index_m - 1
      } else if (bins(index_feature)(index_m).high_split.threshold < value) {
        index_l = index_m + 1
      } else {
        return index_m
      }
    }
    -1
  }
}