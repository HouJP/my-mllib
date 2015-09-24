package bda.local.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.DTreeStrategy
import bda.local.ml.loss.SquaredError
import bda.local.ml.util.Log

/**
 * Decision tree model for classification or regression.
 * This model stores the decision tree structure and parameters.
 *
 * @param topNode root node of decision tree structure
 * @param dTreeStrategy strategy of decision tree
 */
class DTreeModel(
    val topNode: Node,
    val dTreeStrategy: DTreeStrategy) {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features feature vector of a single data point
   * @return predicted value from the trained model
   */
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

  /**
   * Predict values for the given data using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input Array of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points
   * @return Array stored prediction
   */
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

  /**
   * Store decision tree model on the disk.
   *
   * @param path path of the location on the disk
   */
  def save(path: String): Unit = {
    DTreeModel.save(path, this)
  }
}

object DTreeModel {

  /**
   * Store decision tree model on the disk.
   *
   * @param path path of the location on the disk
   * @param model decision tree model
   */
  def save(path: String, model: DTreeModel): Unit = {
    // TODO
  }

  /**
   * Load decision tree model from the disk.
   *
   * @param path path of the localtion on the disk
   */
  def load(path: String): Unit = {
    // TODO
  }
}