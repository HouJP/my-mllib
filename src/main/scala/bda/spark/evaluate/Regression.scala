package bda.spark.evaluate

import org.apache.spark.rdd.RDD

/**
 * RMSE evaluation
 */
object Regression {

  /**
   * Evaluate RMSE of predictions
   *
   * @param tys  RDD[(trueValue, prediction)]
   * @return
   */
  def RMSE(tys: RDD[(Double, Double)]): Double = {
    val s = tys.map {
      case (t, y) => (math.pow((t - y), 2), 1)
    }.reduce { (a, b) =>
      (a._1 + b._1, a._2 + b._2)
    }

    math.sqrt(s._1 / s._2)
  }
}
