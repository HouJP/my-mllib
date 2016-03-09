package bda.spark.model.tree.cart.impurity

import bda.spark.model.tree.cart.CARTPoint
import org.apache.spark.rdd.RDD

private[cart] trait Impurity extends Serializable {

  def stat(cart_ps: RDD[CARTPoint]): ImpurityStatus

  def calculate(stat: ImpurityStatus): Double

  def calculate_weighted(l_count: Double,
                         r_count: Double,
                         l_impurity: Double,
                         r_impurity: Double): Double = {

    val tol = l_count + r_count

    if ((1.0 - 1e-6 <= l_impurity) || (1.0 - 1e-6 <= r_impurity)) {
      1.0
    } else {
      l_impurity * l_count / tol + r_impurity * r_count / tol
    }
  }

  def predict(stat: ImpurityStatus): Double

  def agg(stat: ImpurityStatus,
          num_bins: Array[Int]): ImpurityAggregator
}