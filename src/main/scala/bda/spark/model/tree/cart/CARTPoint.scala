package bda.spark.model.tree.cart

import bda.common.obj.LabeledPoint
import bda.common.util.PoissonDistribution
import bda.spark.model.tree.FeatureSplit
import org.apache.spark.rdd.RDD

/**
  * Case class which stored label, weight and binned features of a data point.
  *
  * @param label      the label of a data point
  * @param weight     the weight of a data point
  * @param binned_fs  binned features of a data point
  */
case class CARTPoint(label: Double,
                     weight: Int,
                     binned_fs: Array[Int]) extends Serializable {

  /**
    * Method to convert the point to an instance of [[String]].
    *
    * @return an instance of [[String]] which represent the point
    */
  override def toString = {
    s"label($label), weight($weight), binned_fs(${binned_fs.mkString(",")})"
  }
}

/**
  * Static methods of [[CARTPoint]].
  */
object CARTPoint {

  /**
    * Method to convert training data set to a RDD of [[CARTPoint]].
    *
    * @param lps      training data set, represented as a RDD of [[LabeledPoint]]
    * @param splits   an array of [[FeatureSplit]] of all features
    * @param n_fs     number of features
    * @param row_rate sampling ratio of training data set
    * @return
    */
  def toCARTPoint(lps: RDD[LabeledPoint],
                  splits: Array[Array[FeatureSplit]],
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

  /**
    * Method to convert a instance of [[LabeledPoint]] to a instance of [[CARTPoint]].
    *
    * @param lp     an instance of [[LabeledPoint]]
    * @param splits an array of [[FeatureSplit]] of all features
    * @param n_fs   number of features
    * @param ps     an instance of [[PoissonDistribution]] used to set weight of the point
    * @return       an instance of [[CARTPoint]]
    */
  def toCARTPoint(lp: LabeledPoint,
                  splits: Array[Array[FeatureSplit]],
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