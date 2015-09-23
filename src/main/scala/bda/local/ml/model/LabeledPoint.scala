package bda.local.ml.model

import bda.common.linalg.immutable.SparseVector

/**
 * Class that represents the features and labels of a data point.
 * @param label Label of the data point.
 * @param features Features of the data point.
 */
case class LabeledPoint(label: Double, features: SparseVector[Double]) {
  override  def toString: String = {
    s"($label,$features)"
  }
}
