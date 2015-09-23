package bda.local.ml.util

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.model.LabeledPoint

import scala.io.Source

object MLUtils {

  /**
   * Loads file into an Array[String]
   * @param path file path in local file system URI
   * @return file content stored as an Array[String]
   */
  def loadFile(path: String): Array[String] = {
    val content = collection.mutable.ArrayBuffer[String]()
    Source.fromFile(path).getLines().foreach(content.append(_))

    content.toArray
  }

  def loadLibSVMFile(path: String): Array[LabeledPoint] = {
    val parsed = loadFile(path)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split(' ')
        val label = items.head.toDouble
        val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
          val value = indexAndValue(1).toDouble
          (index, value)
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