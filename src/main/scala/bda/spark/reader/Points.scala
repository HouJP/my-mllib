package bda.spark.reader

import bda.common.Logging
import bda.common.linalg.immutable.SparseVector
import bda.common.obj.{LabeledPoint, SVDFeaturePoint}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * Read labeledPoints
 */
object Points extends Logging {

  /**
   * Parse LabeledPoint from libsvm format RDD Strings.
   * Each line is a labeled point (libsvm datasets do not have unlabeled data):
   * label fid:v fid:v ...
   *
   * where
   * - Label is {-1, +1} for binary classification, and {0, ..., K-1} for
   * multi-classification.
   * - Fid is start from 1, which should subtract 1 to 0-started.
   * - v is a Double
   *
   * @note User have to specify the feature number ahead
   *
   * @param pt  Input file path
   */
  def readLibSVMFile(sc: SparkContext,
                     pt: String): RDD[LabeledPoint] = {
    // parse the LibSVM file
    val rds: RDD[(Double, Array[(Int, Double)])] = sc.textFile(pt).map { ln =>
      val items = ln.split(" ")
      val label = items(0).toDouble

      // decrease fid by 1
      val fvs = items.tail.map { fv =>
        val Array(fid, v) = fv.split(":")
        (fid.toInt - 1, v.toDouble)
      }
      (label, fvs)
    }

    val n_label = rds.map(_._1).distinct().count().toInt
    val n_feature = rds.map(_._2.map(_._1).max).max + 1
    logInfo(s"n(label)=${n_label}, n(feature)=${n_feature}")

    // transform to labeled points, and adjust label
    rds.map { case (label, fvs) =>
      val new_label: Double = if (n_label > 2) {
        // for multi-class, decrease label to [0, C-1)
        label - 1
      } else {
        // for binary class
        if (label < 0) 0.0 else label
      }
      val fs = SparseVector(n_feature, fvs)
      LabeledPoint(new_label, fs)
    }
  }
}
