package bda.spark.model.tree.gbdt.impurity

import bda.spark.model.tree.gbdt.CARTPoint

/**
  * Class of aggregator to statistic status of labels for computing of Variance impurity.
  *
  * @param off_fs an array recording shift of features
  * @param n_feature number of features
  * @param n_bins an array recording number of bins for features
  * @param stats an array stored status of labels
  * @param step length of status for each bins in [[stats]]
  */
private[gbdt] class VarianceAggregator(off_fs: Array[Int],
                                       n_feature: Int,
                                       n_bins: Array[Int],
                                       stats: Array[Double],
                                       step: Int) extends ImpurityAggregator(off_fs,n_feature, n_bins, stats, step) {

  /**
    * Method to update status array with specified point and feature.
    *
    * @param label label of specified data point
    * @param id_f ID of specified feature, indexed from 0
    * @param binned_f Bin-ID of specified feature, indexed from 0
    */
  def update(label: Double, id_f: Int, binned_f: Int): Unit = {
    val off_b = off_fs(id_f) + step * binned_f
    val absolute_label = math.abs(label)
    stats(off_b) += 1
    stats(off_b + 1) += label
    stats(off_b + 2) += label * label
    stats(off_b + 3) += absolute_label * (1.0 - absolute_label)
  }

  /**
    * Method to update status array with specified point and sub features.
    *
    * @param point the data point used to update array of status
    * @param id_label ID of the label
    */
  def update(point: CARTPoint, id_label: Int) = {
    Range(0, n_feature).foreach {
      id_f =>
        update(point.y_K(id_label), id_f, point.binned_fs(id_f))
    }
  }

  /**
    * Method to calculate Variance impurity, prediction and labels count for left child.
    *
    * @param id_f ID of the feature specified, indexed from 0
    * @param id_s ID of the [[bda.spark.model.tree.FeatureSplit]] specified, indexed from 0
    * @param num_label number of different labels
    * @return (impurity, prediction, labels count) of left child
    */
  def calLeftInfo(id_f: Int, id_s: Int, num_label: Int): (Double, Double, Double) = {
    val off = off_fs(id_f) + step * id_s

    val sub_stat = stats.slice(off, off + step)
    val total = sub_stat(0)
    val impurity = Variance.calculate(sub_stat)
    val predict = Variance.predict(sub_stat, num_label)

    (impurity, predict, total)
  }

  /**
    * Method to calculate Gini impurity, prediction and labels count for right child.
    *
    * @param id_f ID of the feature specified, indexed from 0
    * @param id_s ID of the [[bda.spark.model.tree.FeatureSplit]] specified, indexed from 0
    * @param num_label number of different labels
    * @return (impurity, prediction, labels count) of right child
    */
  def calRightInfo(id_f: Int, id_s: Int, num_label: Int): (Double, Double, Double) = {
    val off = off_fs(id_f) + step * id_s
    val off_last = off_fs(id_f) + step * (n_bins(id_f) - 1)

    val sub_stat = Range(0, step).map {
      id =>
        stats(off_last + id) - stats(off + id)
    }.toArray
    val total = sub_stat(0)
    val impurity = Variance.calculate(sub_stat)
    val predict = Variance.predict(sub_stat, num_label)

    (impurity, predict, total)
  }
}