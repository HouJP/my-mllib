package mymllib.spark.model.tree.cart.impurity

import mymllib.spark.model.tree.cart.CARTPoint
import org.apache.spark.rdd.RDD

/**
  * Class of Gini Impurity for Decision Trees, used for classification.
  */
private[cart] object Gini extends Impurity {

  /**
    * Statistic basic information for calculating impurity.
    *     For Gini: number of nodes, distinct numbers of labels, map for labels, number of labels.
    *
    * @param cart_ps RDD of [[CARTPoint]]
    * @return an instance of [[ImpurityStatus]]
    */
  def stat(cart_ps: RDD[CARTPoint]): ImpurityStatus = {
    // Statistic the number of data points
    val num = cart_ps.map(_.weight).sum.toInt
    // Statistic the distinct number of labels
    val cnt = cart_ps.map(p => (p.label, p.weight.toDouble)).reduceByKey(_+_).collectAsMap()
    val stt = cnt.values.toArray
    // map for labels
    val map_label = cnt.keys.zipWithIndex.toMap
    // number of labels
    val num_label = stt.length
    new GiniStatus(num, stt, map_label, num_label)
  }

  /**
    * Calculate impurity according to [[ImpurityStatus]].
    *     For Gini: 1.0 - sum(rate(i) * rate(i))
    *
    * @param stat an instance of [[ImpurityStatus]]
    * @return value of impurity
    */
  def calculate(stat: ImpurityStatus): Double = {
    if (0 == stat.num) {
      // It's a bad case if nodes has no data points,
      //  and we set impurity=1.0 to prevent this case.
      1.0
    } else {
      1.0 - stat.stt.map {
        num =>
          math.pow(num / stat.num, 2.0)
      }.sum
    }
  }

  /**
    * Calculate impurity according to an array recording status.
    *     For Gini: 1.0 - sum(rate(i) * rate(i))
    *
    * @param stat an array recording status
    * @return value of impurity
    */
  def calculate(stat: Array[Double]): Double = {
    // The len(stat) must greater than 2, including number of labels and
    //    distinct number of labels(at least 2 different labels).
    require(stat.length > 2,
      "require length(status step) > 2 when calculate gini impurity.")

    val tol = stat(0)
    if (0 == tol) {
      // It's a bad case if nodes has no data points,
      //  and we set impurity=1.0 to prevent this case.
      1.0
    } else {
      1.0 - Range(1, stat.length).map {
        id =>
          math.pow(stat(id) / tol, 2)
      }.sum
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

    if ((1.0 - 1e-6 <= l_impurity) ||
      (1.0 - 1e-6 <= r_impurity) ||
      (l_count < min_node_size) ||
      (r_count < min_node_size)) {
      1.0
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
    val real_stat = stat.asInstanceOf[GiniStatus]
    real_stat.map_label.mapValues {
      id =>
        real_stat.stt(id)
    }.maxBy(_._2)._1
  }

  /**
    * Predict the value for a node.
    *
    * @param stat an array recording status.
    * @param map_label a map for label to position
    * @return the prediction of the node.
    */
  def predict(stat: Array[Double], map_label: Map[Double, Int]): Double = {
    map_label.map {
      case (label, pos) =>
        (label, stat(pos + 1))
    }.maxBy(_._2)._1
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

    val gini_stat = stat.asInstanceOf[GiniStatus]
    val step = stat.stt.length + 1
    val off_fs = n_bins.scanLeft(0)((sum, n_bin) => sum + n_bin * step)
    val tol_bins = n_bins.sum
    val stt = Array.fill[Double](tol_bins * step)(0.0)

    new GiniAggregator(off_fs, n_bins, stt, step, gini_stat.map_label)
  }
}