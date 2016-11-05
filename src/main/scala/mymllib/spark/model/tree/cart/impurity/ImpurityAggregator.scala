package mymllib.spark.model.tree.cart.impurity

import mymllib.spark.model.tree.cart.CARTPoint

/**
  * Class of aggregator to statistic status of labels for computing of impurity.
  *
  * @param off_fs an array recording shift of features
  * @param n_bins an array recording number of bins for features
  * @param stats an array stored status of labels
  * @param step length of status for each bins in [[stats]]
  */
private[cart] abstract class ImpurityAggregator(val off_fs: Array[Int],
                                                val n_bins: Array[Int],
                                                val stats: Array[Double],
                                                val step: Int) extends Serializable {

  /**
    * Method to update status array with specified point and sub features.
    *
    * @param point the data point used to update array of status
    * @param sub_fs specified sub features while updating
    */
  def update(point: CARTPoint, sub_fs: Array[Int])

  /**
    * Method to calculate impurity, prediction and labels count for left child.
    *
    * @param id_f ID of the feature specified, indexed from 0
    * @param id_s ID of the [[mymllib.spark.model.tree.FeatureSplit]] specified, indexed from 0
    * @return (impurity, prediction, labels count) of left child
    */
  def calLeftInfo(id_f: Int, id_s: Int): (Double, Double, Double)

  /**
    * Method to calculate impurity, prediction and labels count for right child.
 *
    * @param id_f ID of the feature specified, indexed from 0
    * @param id_s ID of the [[mymllib.spark.model.tree.FeatureSplit]] specified, indexed from 0
    * @return (impurity, prediction, labels count) of right child
    */
  def calRightInfo(id_f: Int, id_s: Int): (Double, Double, Double)

  /**
    * Method to merge another aggregator into this aggregator.
    *
    * @param other another [[ImpurityAggregator]]
    * @return this aggregator after merging
    */
  def merge(other: ImpurityAggregator): ImpurityAggregator = {
    stats.indices.foreach {
      id =>
        stats(id) += other.stats(id)
    }
    this
  }

  /**
    * Method to convert value this status in [[stats]] into prefix summation form.
    *
    * @return this aggregator after convertion
    */
  def toPrefixSum: ImpurityAggregator = {
    n_bins.indices.foreach {
      id_f =>
        Range(off_fs(id_f) + step, off_fs(id_f) + n_bins(id_f) * step).foreach {
          id =>
            stats(id) += stats(id - step)
        }
    }
    this
  }

  /**
    * Convert this class into a [[String]].
    *
    * @return a [[String]] represented this instance of [[ImpurityAggregator]]
    */
  override def toString = {
    s"off_set(${off_fs.mkString(",")}), stats(${stats.mkString(",")})"
  }
}