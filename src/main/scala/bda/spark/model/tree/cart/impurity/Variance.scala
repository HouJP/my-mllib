package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint
import org.apache.spark.rdd.RDD

/**
  * Class of Variance Impurity for Decision Trees, used for classification.
  */
private[cart] object Variance extends Impurity {

  /**
    * Statistic basic information for calculating impurity.
    *     For Variance: number of nodes, summation of labels, squared summation of labels.
    *
    * @param cart_ps RDD of [[CARTPoint]]
    * @return an instance of [[ImpurityStatus]]
    */
  def stat(cart_ps: RDD[CARTPoint]): ImpurityStatus = {
    // Statistic the number of data points
    val n = cart_ps.map(_.weight).sum.toInt
    // Statistic the sum of labels
    val sum = cart_ps.map(p => p.label * p.weight).sum
    // Statistic the squared sum of the labels
    val squared_sum = cart_ps.map(p => p.label * p.label * p.weight).sum
    val stt = Array(sum, squared_sum)
    new VarianceStatus(n, stt)
  }

  /**
    * Calculate impurity according to [[ImpurityStatus]].
    *     For Variance: 1.0 / n * (sum(y(i) * y(i)) - 1.0 / n * (sum(y(i)) * sum(y(i))))
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return value of impurity
    */
  def calculate(stat: ImpurityStatus): Double = {
    if (0 == stat.num) {
      // It's a bad case if nodes has no data points,
      //  and we set impurity=Double.MaxValue/10.0 to prevent this case.
      Double.MaxValue / 10.0
    } else {
      (stat.stt(1) - math.pow(stat.stt(0), 2) / stat.num) / stat.num
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
    // The len(stat) must be 3, including number of labels and
    //    summation of labels and squared summation of labels.
    require(stat.length == 3,
      "require length(status step) == 3 when calculate variance impurity.")

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
    * @param l_count number of left child
    * @param r_count number of right child
    * @param l_impurity impurity of left child
    * @param r_impurity impurity of right child
    * @return the weighted impurity
    */
  def calculate_weighted(l_count: Double,
                         r_count: Double,
                         l_impurity: Double,
                         r_impurity: Double): Double = {

    val tol = l_count + r_count

    if ((Double.MaxValue / 10.0 - 1e-6 <= l_impurity) ||
      (Double.MaxValue / 10.0 - 1e-6 <= r_impurity)) {
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
  def predict(stat: ImpurityStatus): Double = {
    if (0 == stat.num) {
      0.0
    } else {
      stat.stt(0) / stat.num
    }
  }

  /**
    * Predict the value for a node.
    *
    * @param stat an array recording status.
    * @return the prediction of the node.
    */
  def predict(stat: Array[Double]): Double = {
    // The len(stat) must be 3, including number of labels and
    //    summation of labels and squared summation of labels.
    require(stat.length == 3,
      "require length(status step) == 3 when calculate variance prediction.")

    if (0 == stat(0)) {
      0.0
    } else {
      stat(1) / stat(0)
    }
  }

  /**
    * Generate an instance of [[ImpurityAggregator]] according to the impurity status and numbers of bins.
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @param n_bins an array which recorded numbers of bins for features
    * @return an instance of [[ImpurityStatus]]
    */
  def agg(stat: ImpurityStatus,
          n_bins: Array[Int]): ImpurityAggregator = {

    val step = stat.stt.length + 1
    val off_fs = n_bins.scanLeft(0)((sum, n_bin) => sum + n_bin * step)
    val tol_bins = n_bins.sum
    val stt = Array.fill[Double](tol_bins * step)(0.0)

    new VarianceAggregator(off_fs, n_bins, stt, step)
  }
}