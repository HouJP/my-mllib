package bda.local.ml.para

import bda.local.ml.para.Loss._
import bda.local.ml.loss.SquaredErrorCalculator

/**
 * Class of strategy for GBoost.
 *
 * @param num_iter number of iterations
 * @param learn_rate vlaue of learning rate
 * @param loss loss type [[bda.local.ml.para.Loss]]
 * @param min_step minimum step of each iteration, or stop it
 */
class GBoostPara (
    val dt_para: DTreePara = DTreePara.default,
    val num_iter: Int = 200,
    val learn_rate: Double = 0.2,
    val loss: Loss = SquaredError,
    val min_step: Double = 1e-5) {

  val loss_calculator = loss match {
    case SquaredError => SquaredErrorCalculator
    case _ => throw new IllegalArgumentException(s"Did not recognize loss type: ${loss}")
  }
}

object GBoostPara {

  /**
   * Construct a [[bda.local.ml.para.GBoostPara]] instance using default parameters
   *
   * @return a [[bda.local.ml.para.GBoostPara]] instance
   */
  def default(): GBoostPara = {
    new GBoostPara()
  }
}