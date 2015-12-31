package bda.local.reader

import bda.common.Logging
import bda.common.linalg.immutable.SparseVector
import bda.common.obj.{LabeledPoint, SVDFeaturePoint}
import bda.common.util.io.readLines

/**
 * Read Points
 */
object Points extends Logging {

  /**
   * Read a LibSVM File with format:
   * Each line is a labeled point (libsvm dataset do not have unlabeled data):
   * label fid:v fid:v ...
   *
   * where
   * - Label is {-1, +1} for binary classification, and {1, ..., K} for
   * multi-classification.
   * - Fid is start from 1, which should subtract 1 to 0-started.
   * - v is a Double
   *
   * @note User have to specify the feature number ahead
   *
   * @param pt  Input file path
   */
  def readLibSVMFile(pt: String): Seq[LabeledPoint] = {
    var n_feature = 0
    // parse the LibSVM file
    val rds = readLines(pt).map { ln =>
      val items = ln.split(" ")
      // decrease fid by 1
      val fvs = items.tail.map { fv =>
        val Array(fid, v) = fv.split(":")
        val f = fid.toInt
        n_feature = math.max(f, n_feature)
        (f - 1, v.toDouble)
      }
      val label = items(0).toDouble
      (label, fvs)
    }.toSeq

    // determine the class number
    val n_label = rds.map(_._1).distinct.size
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