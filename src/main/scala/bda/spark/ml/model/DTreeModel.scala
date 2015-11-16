package bda.spark.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.spark.ml.loss.{SquaredErrorCounter}
import bda.spark.ml.para.DTreePara
import bda.local.ml.util.Log
import bda.local.ml.model.LabeledPoint
import bda.spark.ml.para.Loss._
import org.apache.spark.rdd.RDD

/**
 * Decision tree model for classification or regression.
 * This model stores the decision tree structure and parameters.
 *
 * @param root root node of decision tree structure
 * @param dt_para strategy of decision tree
 */
class DTreeModel(
    val root: Node,
    val dt_para: DTreePara) extends Serializable {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point
   * @return predicted value from the trained model
   */
  def predict(fs: SparseVector[Double]): Double = {
    var node = root
    while (!node.is_leaf) {
      if (fs(node.split.get.feature) <= node.split.get.threshold) {
        node = node.left_child.get
      } else {
        node = node.right_child.get
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
  def predict(input: RDD[LabeledPoint]): RDD[Double] = {
    val loss = dt_para.loss_calculator

    val pred_err = input.map { case lp =>
      val pred = predict(lp.features)
      val err = loss.computeError(pred, lp.label)
      (pred, err)
    }

    val err = pred_err.values.mean()

    Log.log("INFO", s"predict done, with RMSE = ${math.sqrt(err)}")

    pred_err.keys
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