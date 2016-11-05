package mymllib.spark.model.tree.gbdt

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.LabeledPoint
import mymllib.spark.model.tree.FeatureSplit
import org.apache.spark.rdd.RDD

/**
  * Case class which stored labels, F-values, features and binned features of a data point.
  *
  * @param label     label of the data point
  * @param f_K       f-values of K classes
  * @param fs        features of the data point
  * @param binned_fs binned features of the data point
  */
private[gbdt] case class GBDTPoint(id: String,
                                   label: Int,
                                   f_K: Array[Double],
                                   fs: SparseVector[Double],
                                   binned_fs: Array[Int]) extends Serializable {

  /**
    * Method to convert the point to an instance of [[String]].
    *
    * @return an instance of [[String]] which represent the point
    */
  override def toString = {
    s"id($id),label($label),f_K(${f_K.mkString(",")}),binned_fs(${binned_fs.mkString(",")})"
  }
}

/**
  * Static methods of [[GBDTPoint]].
  */
private[gbdt] object GBDTPoint {

  /**
    * Method to convert data set to a RDD of [[GBDTPoint]].
    *
    * @param lps       data set which represented as [[GBDTPoint]]
    * @param splits    splits for all features
    * @param n_label   number of labels
    * @param n_feature number of features
    * @return a RDD of [[GBDTPoint]]
    */
  def toGBDTPoint(lps: RDD[LabeledPoint],
                  splits: Array[Array[FeatureSplit]],
                  n_label: Int,
                  n_feature: Int): RDD[GBDTPoint] = {
    lps.map {
      lp =>
        GBDTPoint.toGBDTPoint(lp, splits, n_label, n_feature)
    }
  }

  /**
    * Method to convert a data point to an instance of [[GBDTPoint]].
    *
    * @param lp        the data point which represented as [[LabeledPoint]]
    * @param splits    splits for all features
    * @param n_label   number of labels
    * @param n_feature number of features
    * @return a instance of [[GBDTPoint]]
    */
  def toGBDTPoint(lp: LabeledPoint,
                  splits: Array[Array[FeatureSplit]],
                  n_label: Int,
                  n_feature: Int): GBDTPoint = {

    val binned_fs = new Array[Int](n_feature)
    Range(0, n_feature).foreach {
      id_f =>
        val binned_f = binarySearchForBin(lp.fs(id_f), splits, id_f)
        binned_fs(id_f) = binned_f
    }

    new GBDTPoint(lp.id, lp.label.toInt, Array.fill[Double](n_label)(0.0), lp.fs, binned_fs)
  }

  /**
    * Method to find bin-id by binary search with specified feature-value and feature-id.
    *
    * @param v      value of specified point and feature
    * @param splits an array of [[FeatureSplit]] of all features
    * @param id_f   ID of specified feature
    * @return       ID of bin with specified point and feature
    */
  def binarySearchForBin(v: Double,
                         splits: Array[Array[FeatureSplit]],
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