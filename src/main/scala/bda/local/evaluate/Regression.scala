package bda.local.evaluate

/**
 * RMSE is the Root-Mean-Square Error between true values and
 * predicted values.
 */
object Regression {

  /**
   * Compute the RMSE of predicted values
   * @param tys  sequence of true-predicted value pairs, Seq[(true_v, predicted_v)]
   * @return RMSE
   */
  def RMSE(tys: Seq[(Double, Double)]): Double = {
    assert(tys != null && tys.nonEmpty, "tys in RMSE is empty!")
    val tmp_v = tys.map {
      case (t, y) =>
        val s = (t - y) * (t - y)
        s
    }.sum / tys.size
    math.sqrt(tmp_v)
  }
}
