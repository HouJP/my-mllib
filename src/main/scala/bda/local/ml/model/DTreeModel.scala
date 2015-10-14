package bda.local.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.loss.{SquaredErrorCounter}
import bda.local.ml.para.DTreePara
import bda.local.ml.util.Log
import bda.local.ml.para.Loss._

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
    while (!node.isLeaf) {
      if (fs(node.featureID) < node.splitValue) {
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
    val (pred, err) = computePredictAndError(input, 1.0)

    Log.log("INFO", s"predict done, with mean error = ${err}")

    pred
  }

  /**
   * Store decision tree model on the disk.
   *
   * @param path path of the location on the disk
   */
  def save(path: String): Unit = {
    DTreeModel.save(path, this)
  }

  /**
   * Predict values and get the mean error for the given data using the model trained and the model weight
   *
   * @param input Array of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points
   * @param weight model weight
   * @return Array stored prediction and the mean error
   */
  def computePredictAndError(
      input: Array[LabeledPoint],
      weight: Double): (Array[Double], Double) = {
    val err_counter = dt_para.loss match {
      case SquaredError => new SquaredErrorCounter()
      case _ => throw new IllegalArgumentException(s"Did not recognize loss type: ${dt_para.loss}")
    }

    val pred = input.map { lp =>
      val pred = predict(lp.features) * weight
      err_counter :+= (pred, lp.label)
      pred
    }

    (pred, err_counter.getMean)
  }

  /**
   * Update the pre-predictions and get the mean error for the given data using the model trained and the model weight
   *
   * @param input Array of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points
   * @param pre_pred pre-predictions
   * @param weight model weight
   * @return Array stored prediction and the mean error
   */
  def updatePredictAndError(
      input: Array[LabeledPoint],
      pre_pred: Array[Double],
      weight: Double): (Array[Double], Double) = {
    val err_counter = dt_para.loss match {
      case SquaredError => new SquaredErrorCounter()
      case _ => throw new IllegalArgumentException(s"Did not recognize loss type: ${dt_para.loss}")
    }

    val pred = input.zip(pre_pred).map { case (lp, pre_pred) =>
      val pred = pre_pred + predict(lp.features) * weight
      err_counter :+= (pred, lp.label)
      pred
    }

    (pred, err_counter.getMean)
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