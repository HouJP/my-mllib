package mymllib.spark.model.tree.gbdt.impurity

import mymllib.spark.model.tree.gbdt.CARTPoint

/**
  * Class of aggregator to statistic status of labels for computing of impurity.
  *
  * @param off_fs an array recording shift of features
  * @param n_feature number of features
  * @param n_bins an array recording number of bins for features
  * @param stats an array stored status of labels
  * @param step length of status for each bins in [[stats]]
  */
private[gbdt] abstract class ImpurityAggregator(val off_fs: Array[Int],
                                                val n_feature: Int,
                                                val n_bins: Array[Int],
                                                val stats: Array[Double],
                                                val step: Int) extends Serializable {

  /**
    * Convert this class to a [[String]].
    *
    * @return a [[String]] represented this instance of [[ImpurityAggregator]]
    */
  override def toString = {
    s"off_set(${off_fs.mkString(",")}),n_feature($n_feature),stats(${stats.mkString(",")})"
  }

  /**
    * Method to update status array with specified point and sub features.
    *
    * @param point the data point used to update array of status
    * @param id_label ID of the label
    */
  def update(point: CARTPoint, id_label: Int)

  /**
    * Method to calculate impurity, prediction and labels count for left child.
    *
    * @param id_f      ID of the feature specified, indexed from 0
    * @param id_s      ID of the [[mymllib.spark.model.tree.FeatureSplit]] specified, indexed from 0
    * @param num_label number of different labels
    * @return (impurity, prediction, labels count) of left child
    */
  def calLeftInfo(id_f: Int, id_s: Int, num_label: Int): (Double, Double, Double)

  /**
    * Method to calculate impurity, prediction and labels count for right child.
 *
    * @param id_f      ID of the feature specified, indexed from 0
    * @param id_s      ID of the [[mymllib.spark.model.tree.FeatureSplit]] specified, indexed from 0
    * @param num_label number of different labels
    * @return (impurity, prediction, labels count) of right child
    */
  def calRightInfo(id_f: Int, id_s: Int, num_label: Int): (Double, Double, Double)

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
    * Method to convert value of [[stats]] into prefix summation form.
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
}