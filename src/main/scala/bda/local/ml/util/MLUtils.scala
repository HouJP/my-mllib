package bda.local.ml.util

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.model.LabeledPoint

import scala.io.Source

/**
 * Class of tools used to load data.
 */
object MLUtils {

  /**
   * Loads file into an Array[String]
   *
   * @param path file path in local file system URI
   * @return file content stored as an Array[String]
   */
  def loadFile(path: String): Array[String] = {
    val c = collection.mutable.ArrayBuffer[String]()
    Source.fromFile(path).getLines().foreach(c.append(_))

    c.toArray
  }

  /**
   * Loads labeled data in the LIBSVM format into an Array[LabeledPoint].
   *
   * @param path the path of the datafile on the disk
   * @return labeled data stored as an Array[LabeledPoint]
   */
  def loadLibSVMFile(path: String): Array[LabeledPoint] = {
    val parsed = loadFile(path)
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