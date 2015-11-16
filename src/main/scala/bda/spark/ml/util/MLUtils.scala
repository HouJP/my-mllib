package bda.spark.ml.util

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.model.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source

/**
 * Class of tools used to load data.
 */
object MLUtils {

  /**
   * Loads labeled data in the LIBSVM format into an RDD[[bda.local.ml.model.LabeledPoint]]
   * @param sc Spark context
   * @param path file path in any Hadoop-supported file system URI
   * @return labeled data stored as an RDD[[bda.local.ml.model.LabeledPoint]]
   */
  def loadLibSVMFile(
      sc: SparkContext,
      path: String): RDD[LabeledPoint] = {
    val parsed = sc.textFile(path, sc.defaultMinPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      val items = line.split(' ')
      val label = items.head.toDouble
      val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
        val ind_val = item.split(':')
        val ind = ind_val(0).toInt - 1 // Convert 1-based indices to 0-based.
      val value = ind_val(1).toDouble
        (ind, value)
      }.unzip
      (label, indices.toArray, values.toArray)
    }

    // Determine number of features.
    val n = parsed.map { case (label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1

    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, new SparseVector[Double](n, indices, values))
    }
  }
}