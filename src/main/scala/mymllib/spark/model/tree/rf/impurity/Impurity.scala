package mymllib.spark.model.tree.rf.impurity

import bda.common.linalg.immutable.SparseVector
import mymllib.spark.model.tree.TreeNode


/**
  * Trait for impurity of Decision Trees.
  */
private[rf] trait Impurity extends Serializable {


  /**
    * Predict value for the node.
    *
    * @param fs feature vector of single data point
    * @param wk_learners weak learners which formed the model
    * @return the prediction according to the impurity status
    */
  def predict(fs: SparseVector[Double], wk_learners: Array[TreeNode]): Double

}