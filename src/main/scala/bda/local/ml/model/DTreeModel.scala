package bda.local.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.DTreeStrategy
import bda.local.ml.loss.SquaredError
import bda.local.ml.util.Log

class DTreeModel(
    val topNode: Node,
    val dTreeStrategy: DTreeStrategy) {

  def predict(features: SparseVector[Double]): Double = {
    var node = topNode
    while (!node.isLeaf) {
      if (features(node.featureID) < node.splitValue) {
        node = node.leftNode.get
      } else {
        node = node.rightNode.get
      }
    }
    node.predict
  }

  def predict(input: Array[LabeledPoint]): Array[Double] = {
    val se = new SquaredError

    val output = input.map { p =>
      val pre = predict(p.features)
      se :+ (pre, p.label)
      pre
    }

    Log.log("INFO", s"predict done, with RMSE = ${se.getRMSE}")

    output
  }

  def save(path: String): Unit = {
    DTreeModel.save(path, this)
  }
}

object DTreeModel {

  def save(path: String, model: DTreeModel): Unit = {

  }

  def load(path: String): Unit = {

  }
}