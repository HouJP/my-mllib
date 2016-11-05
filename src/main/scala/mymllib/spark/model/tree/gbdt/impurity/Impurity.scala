package mymllib.spark.model.tree.gbdt.impurity

import mymllib.spark.model.tree.gbdt.CARTPoint
import org.apache.spark.rdd.RDD

/**
  * Trait for impurity of Decision Trees.
  */
private[gbdt] trait Impurity extends Serializable {

  /**
    * Statistic basic information for calculating impurity.
    *     For Variance: number of different labels, number of total labels, summation of labels, squared summation of labels.
    *
    * @param cart_ps RDD of [[CARTPoint]]
    * @return an instance of [[ImpurityStatus]]
    */
  def stat(n_label: Int, cart_ps: RDD[CARTPoint]): ImpurityStatus

  /**
    * Calculate impurity according to [[ImpurityStatus]].
    *     For Variance: 1.0 / n * (sum(y(i) * y(i)) - 1.0 / n * (sum(y(i)) * sum(y(i))))
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return value of impurity
    */
  def calculate(stat: ImpurityStatus): Array[Double]

  /**
    * Calculate the weighted impurity according to number and impurity of left and right children.
    *
    * @param l_count       number of left child
    * @param r_count       number of right child
    * @param l_impurity    impurity of left child
    * @param r_impurity    impurity of right child
    * @param min_node_size minimum size of node
    * @return the weighted impurity
    */
  def calculate_weighted(l_count: Double, r_count: Double, l_impurity: Double, r_impurity: Double, min_node_size: Int): Double

  /**
    * Predict the value for a node.
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return the prediction according to the impurity status
    */
  def predict(stat: ImpurityStatus): Array[Double]

  /**
    * Generate an instance of [[ImpurityAggregator]] according to the impurity status and numbers of bins.
    *
    * @param num_feature  number of features
    * @param num_bins     an array which recorded numbers of bins for features
    * @return             an instance of [[ImpurityStatus]]
    */
  def agg(num_feature: Int, num_bins: Array[Int]): ImpurityAggregator
}