package bda.spark.ml.model

import bda.local.ml.model.LabeledPoint
import bda.local.ml.util.Log
import bda.spark.ml.para.DTreePara
import org.apache.spark.rdd.RDD

class DTreeMetaData(
    val num_features: Int,
    val num_examples: Int,
    val max_bins: Int,
    val num_bins: Array[Int]) extends Serializable {

  def numSplits(index_feature: Int): Int = {
    num_bins(index_feature) - 1
  }

  def setBins(index_feature: Int, num_splits: Int): Unit = {
    num_bins(index_feature) = num_splits + 1
  }
}

object DTreeMetaData {

  def build(
      input: RDD[LabeledPoint],
      dt_para: DTreePara): DTreeMetaData = {

    val num_features = input.take(1)(0).features.size
    val num_examples = input.count().toInt
    val max_bins = math.min(dt_para.max_bins, num_examples)
    val num_bins = Array.fill[Int](num_features)(max_bins)

    new DTreeMetaData(num_features, num_examples, max_bins, num_bins)
  }
}