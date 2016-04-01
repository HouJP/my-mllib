package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint
import org.apache.spark.rdd.RDD

/**
  * Trait for impurity of Decision Trees.
  */
private[cart] trait Impurity extends Serializable {

  /**
    * Statistic basic information for calculating impurity.
    *     For Gini: number of nodes, distinct numbers of labels, map for labels, number of labels.
    *     For Variance: number of nodes, summation of labels, squared summation of labels.
    *
    * @param cart_ps RDD of [[CARTPoint]]
    * @return an instance of [[ImpurityStatus]]
    */
  def stat(cart_ps: RDD[CARTPoint]): ImpurityStatus

  /**
    * Calculate impurity according to [[ImpurityStatus]].
    *     For Gini: 1.0 - sum(rate(i) * rate(i))
    *     For Variance: 1.0 / n * (sum(y(i) * y(i)) - 1.0 / n * (sum(y(i)) * sum(y(i))))
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return value of impurity
    */
  def calculate(stat: ImpurityStatus): Double

  /**
    * Calculate the weighted impurity according to number and impurity of left and right children.
    *
    * @param l_count number of left child
    * @param r_count number of right child
    * @param l_impurity impurity of left child
    * @param r_impurity impurity of right child
    * @return the weighted impurity
    */
  def calculate_weighted(l_count: Double, r_count: Double, l_impurity: Double, r_impurity: Double): Double

  /**
    * Predict the value for a node.
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return the prediction according to the impurity status
    */
  def predict(stat: ImpurityStatus): Double

  /**
    * Generate an instance of [[ImpurityAggregator]] according to the impurity status and numbers of bins.
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @param num_bins an array which recorded numbers of bins for features
    * @return an instance of [[ImpurityStatus]]
    */
  def agg(stat: ImpurityStatus,
          num_bins: Array[Int]): ImpurityAggregator
}