package bda.spark.reader

import bda.common.linalg.immutable.SparseVector
import bda.common.obj.{RegPoint, ClassPoint}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * Read a LibSVM File with format:
 * Each line is a labeled point (libsvm datasets do not have unlabeled data):
 * label fid:v fid:v ...
 *
 * where
 * - Label is {-1, +1} for BINARY classification, and {0, ..., K-1} for multi-classification.
 * - Fid is start from 1.
 * - v is a Double
 */
object LibSVMFile {

  /**
   * Read points with specified class number and feature number as [[bda.common.obj.ClassPoint]]
   * @param pt   input file path
   * @param class_num  class number in the file
   * @param feature_num  feature number in the file
   */
  def readAsClass(sc: SparkContext,
                  pt: String,
                  class_num: Int,
                  feature_num: Int): RDD[ClassPoint] = {
    parse(sc, pt).map {
      case (label, fvs) =>
        val t = if (class_num == 2 && label < 0) 0 else label.toInt
        val fs = SparseVector(feature_num, fvs)
        ClassPoint(t, fs)
    }
  }

  /**
   * Read points with specified class number and feature number as [[bda.common.obj.RegPoint]]
   * @param pt   input file path
   * @param class_num  class number in the file
   * @param feature_num  feature number in the file
   */
  def readAsReg(sc: SparkContext,
                pt: String,
                class_num: Int,
                feature_num: Int): RDD[RegPoint] = {
    parse(sc, pt).map {
      case (label, fvs) =>
        val t = label
        val fs = SparseVector(feature_num, fvs)
        RegPoint(t, fs)
    }
  }

  /**
   * Read and determine the class and feature number from data, save as [[bda.common.obj.ClassPoint]]
   * @return  (points, class_num, feature_num)
   */
  def readAsClass(sc: SparkContext,
                  pt: String): (RDD[ClassPoint], Int, Int) = {
    val (class_num, feature_num) = statClassAndFeature(sc, pt)
    val points = readAsClass(sc, pt, class_num, feature_num)
    (points, class_num, feature_num)
  }

  /**
   * Read and determine the feature number from data, save as [[bda.common.obj.RegPoint]]
   * @return  (points, class_num, feature_num)
   */
  def readAsReg(sc: SparkContext,
                pt: String): (RDD[RegPoint], Int) = {
    val (class_num, feature_num) = statClassAndFeature(sc, pt)
    val points = readAsReg(sc, pt, class_num, feature_num)
    (points, feature_num)
  }

  /** Parse each line of the file */
  private def parse(sc: SparkContext,
                    pt: String): RDD[(Double, Array[(Int, Double)])] = {
    sc.textFile(pt).map { ln =>
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
  def statClassAndFeature(sc: SparkContext,
                          pt: String): (Int, Int) = {
    val data = parse(sc, pt)
    val class_num = data.map(_._1).distinct().collect().size
    val feature_num = data.map {
      case (label, fvs) => fvs.map(_._1).max
    }.max + 1
    (class_num, feature_num)
  }
}