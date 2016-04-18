package bda.spark.model.tree.gbdt.impurity

import bda.spark.model.tree.gbdt.CARTPoint
import org.apache.spark.rdd.RDD

/**
  * Class of Variance Impurity for Decision Trees, used for classification.
  */
private[gbdt] object Variance extends Impurity {

  /**
    * Statistic basic information for calculating impurity.
    *     For Variance: number of different labels, number of total lables, summation of labels, squared summation of labels.
    *
    * @param cart_ps RDD of [[CARTPoint]]
    * @return an instance of [[ImpurityStatus]]
    */
  def stat(n_label: Int, cart_ps: RDD[CARTPoint]): ImpurityStatus = {
    // Statistic the number of data points
    val n = cart_ps.count().toInt

    // Statistic the sum, squared sum and sum of absolute product for K labels
    val stt = cart_ps.map {
      e =>
        e.y_K.flatMap {
          y =>
            val squared_y = math.pow(y, 2)
            val absolute_y = math.abs(y)
            val absolute_product_y = absolute_y * (1 - absolute_y)
            Array(y, squared_y, absolute_product_y)
        }
    }.reduce {
      case (a: Array[Double], b: Array[Double]) =>
        a.zip(b).map(e => e._1 + e._2)
    }

    new VarianceStatus(n_label, n, stt)
  }

  /**
    * Calculate impurity according to [[ImpurityStatus]].
    *     For Variance: 1.0 / n * (sum(y(i) * y(i)) - 1.0 / n * (sum(y(i)) * sum(y(i))))
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return value of impurity
    */
  def calculate(stat: ImpurityStatus): Array[Double] = {
    if (0 == stat.num_data) {
      // It's a bad case if nodes has no data points,
      //  and we set impurity=Double.MaxValue/10.0 to prevent this case.
      Array.fill[Double](stat.num_label)(Double.MaxValue / 10.0)
    } else {
      Range(0, stat.num_label).map {
        id =>
          (stat.stt(3 * id + 1) - math.pow(stat.stt(3 * id), 2) / stat.num_data) / stat.num_data
      }.toArray
    }
  }

  /**
    * Calculate impurity according to [[ImpurityStatus]].
    *     For Variance: 1.0 / n * (sum(y(i) * y(i)) - 1.0 / n * (sum(y(i)) * sum(y(i))))
    *
    * @param stat an array recording status
    * @return value of impurity
    */
  def calculate(stat: Array[Double]): Double = {
    // The len(stat) must be 4, including number of labels and
    //    summation of labels, squared summation of labels and absolute product summation of labels.
    require(stat.length == 4,
      "require length(status step) == 4 when calculate variance impurity.")

    if (0 == stat(0)) {
      // It's a bad case if nodes has no data points,
      //  and we set impurity=Double.MaxValue/10.0 to prevent this case.
      Double.MaxValue / 10.0
    } else {
      (stat(2) - math.pow(stat(1), 2) / stat(0)) / stat(0)
    }
  }

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
  def calculate_weighted(l_count: Double,
                         r_count: Double,
                         l_impurity: Double,
                         r_impurity: Double,
                         min_node_size: Int): Double = {

    val tol = l_count + r_count

    if ((Double.MaxValue / 10.0 - 1e-6 <= l_impurity) ||
      (Double.MaxValue / 10.0 - 1e-6 <= r_impurity) ||
      (l_count < min_node_size) ||
      (r_count < min_node_size)) {
      Double.MaxValue / 10.0
    } else {
      l_impurity * l_count / tol + r_impurity * r_count / tol
    }
  }

  /**
    * Predict the value for a node.
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return the prediction according to the impurity status
    */
  def predict(stat: ImpurityStatus): Array[Double] = {
    if (0 == stat.num_data) {
      Array.fill[Double](stat.num_label)(0.0)
    } else {
      Range(0, stat.num_label).map {
        id =>
          stat.stt(3 * id + 0) / stat.stt(3 * id + 2) * (stat.num_label - 1) / stat.num_label
      }.toArray
    }
  }

  /**
    * Predict the value for a node.
    *
    * @param stat an array recording status
    * @param num_label number of different labels
    * @return the prediction of the node
    */
  def predict(stat: Array[Double], num_label: Int): Double = {
    // The len(stat) must be 4, including number of labels and
    //    summation of labels, squared summation of labels and absolute product summation of labels.
    require(stat.length == 4,
      "require length(status step) == 4 when calculate variance prediction.")

    if (0 == stat(0)) {
      0.0
    } else {
      stat(1) / stat(3) * (num_label - 1) / num_label
    }
  }

  /**
    * Generate an instance of [[ImpurityAggregator]] according to the impurity status and numbers of bins.
    *
    * @param n_feature number of features
    * @param n_bins an array which recorded numbers of bins for features
    * @return an instance of [[ImpurityStatus]]
    */
  def agg(n_feature: Int, n_bins: Array[Int]): ImpurityAggregator = {

    // step = 3 + 1
    //    (id = 0): number of labels
    //    (id = 1): summation of labels
    //    (id = 2): squared summation of labels
    //    (id = 3): absolute product summation of labels
    val step = 3 + 1
    val off_fs = n_bins.scanLeft(0)((sum, n_bin) => sum + n_bin * step)
    val tol_bins = n_bins.sum
    val stt = Array.fill[Double](tol_bins * step)(0.0)

    new VarianceAggregator(off_fs, n_feature, n_bins, stt, step)
  }
}