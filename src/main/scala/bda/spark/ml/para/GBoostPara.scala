package bda.spark.ml.para


import bda.spark.ml.para.Loss._
import bda.spark.ml.loss.SquaredErrorCalculator

/**
 * Class of strategy for GBoost.
 * @param dt_para the strategy of decision trees used in gradient boosting.
 * @param num_iter number of iterations
 * @param learn_rate value of learning rate
 * @param loss loss type [[bda.local.ml.para.Loss]]
 * @param min_step minimum step of each iteration, or stop it
 */
class GBoostPara (
    val dt_para: DTreePara = DTreePara.default,
    val num_iter: Int = 300,
    val learn_rate: Double = 0.02,
    val loss: Loss = SquaredError,
    val min_step: Double = 1e-5) extends Serializable {

  val loss_calculator = loss match {
    case SquaredError => SquaredErrorCalculator
    case _ => throw new IllegalArgumentException(s"Did not recognize loss type: $loss")
  }
}

object GBoostPara {

  /**
   * Construct a [[bda.local.ml.para.GBoostPara]] instance using default parameters
   *
   * @return a [[bda.local.ml.para.GBoostPara]] instance
   */
  def default: GBoostPara = {
    new GBoostPara()
  }
}