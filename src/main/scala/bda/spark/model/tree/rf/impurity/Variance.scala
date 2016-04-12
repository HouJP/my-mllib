package bda.spark.model.tree.rf.impurity

import bda.common.linalg.immutable.SparseVector
import bda.spark.model.tree.TreeNode
import bda.spark.model.tree.cart.CARTModel

import scala.collection.mutable

/**
  * Class of Variance Impurity for Random Forest, used for regression.
  */
private[rf] object Variance extends Impurity {

  /**
    * Predict value for the node.
    *
    * @param fs feature vector of single data point
    * @param wk_learners weak learners which formed the Random Forest model
    * @return the prediction according to the impurity status
    */
  def predict(fs: SparseVector[Double], wk_learners: Array[TreeNode]): Double = {
    wk_learners.map {
      wl =>
        CARTModel.predict(fs, wl)
    }.sum / wk_learners.length
  }
}