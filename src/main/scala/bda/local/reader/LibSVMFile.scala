package bda.local.reader

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.{ClassPoint, RegPoint}
import bda.common.util.io.readLines
import scala.collection.mutable.Set

/**
 * Read a LibSVM File with format:
 * Each line is a labeled point (libsvm datasets do not have unlabeled data):
 * label fid:v fid:v ...
 *
 * where
 * - Label is {-1, +1} for binary classification, and {0, ..., K-1} for multi-classification.
 * - Fid is start from 1.
 * - v is a Double
 */
object LibSVMFile {

  /**
   * Read points with specified class number and feature number as [[bda.common.obj.ClassPoint]].
   * @param pt   input file path
   * @param class_num  class number in the file
   * @param feature_num  feature number in the file
   */
  def readAsClass(pt: String, class_num: Int, feature_num: Int): Iterator[ClassPoint] =
    parse(pt).map {
      case (label, fvs) =>
        val t = if (class_num == 2 && label < 0) 0 else label.toInt
        val fs = SparseVector(feature_num, fvs)
        ClassPoint(t, fs)
    }

  /**
   * Read points with specified class number and feature number as [[bda.common.obj.RegPoint]].
   * @param pt   input file path
   * @param class_num  class number in the file
   * @param feature_num  feature number in the file
   */
  def readAsReg(pt: String, class_num: Int, feature_num: Int): Iterator[RegPoint] =
    parse(pt).map {
      case (label, fvs) =>
        val t = label
        val fs = SparseVector(feature_num, fvs)
        RegPoint(t, fs)
    }

  /**
   * Read and determine the class and feature number from data, save as [[bda.common.obj.ClassPoint]].
   * @return  (points, class_num, feature_num)
   */
  def readAsClass(pt: String): (Iterator[ClassPoint], Int, Int) = {
    val (class_num, feature_num) = statClassAndFeature(pt)
    val points = readAsClass(pt, class_num, feature_num)
    (points, class_num, feature_num)
  }

  /**
   * Read and determine the class and feature number from data, save as [[bda.common.obj.RegPoint]].
   * @return  (points, feature_num)
   */
  def readAsReg(pt: String): (Iterator[RegPoint], Int) = {
    val (class_num, feature_num) = statClassAndFeature(pt)
    val points = readAsReg(pt, class_num, feature_num)
    (points, feature_num)
  }

  /** Parse each line of the file */
  private def parse(pt: String): Iterator[(Double, Array[(Int, Double)])] = {
    readLines(pt).map { ln =>
      val items = ln.split(" ")
      val label = items(0).toDouble

      // decrease fid by 1
      val fvs = items.tail.map { fv =>
        val Array(fid, v) = fv.split(":")
        (fid.toInt - 1, v.toDouble)
      }
      (label, fvs)
    }
  }

  /** Count the class number and feature number in the file */
  private def statClassAndFeature(pt: String): (Int, Int) = {
    val labels = Set.empty[Double]
    var max_fid = -1
    parse(pt).foreach {
      case (label, fvs) =>
        labels.add(label)
        max_fid = math.max(max_fid, fvs.map(_._1).max)
    }
    (labels.size, max_fid + 1)
  }
}