package mymllib.spark.model.tree.rf.impurity

import bda.common.linalg.immutable.SparseVector
import mymllib.spark.model.tree.TreeNode
import mymllib.spark.model.tree.cart.CARTModel

import scala.collection.mutable

/**
  * Class of Gini Impurity for Random Forest, used for classification.
  */
private[rf] object Gini extends Impurity {

  /**
    * Predict value for the node.
    *
    * @param fs feature vector of single data point
    * @param wk_learners weak learners which formed the Random Forest model
    * @return the prediction according to the impurity status
    */
  def predict(fs: SparseVector[Double], wk_learners: Array[TreeNode]): Double = {
    val count = mutable.Map[Double, Int]()
    wk_learners.foreach {
      wl =>
        val pred = CARTModel.predict(fs, wl)
        count(pred) = count.getOrElse(pred, 0) + 1
    }
    count.maxBy(_._2)._1
  }
}