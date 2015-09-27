package bda.local.ml.strategy

import bda.local.ml.loss.{Loss, SquaredError}

/**
 * Class of strategy for GBoost.
 *
 * @param numIterations number of iterations
 * @param learningRate vlaue of learning rate
 * @param loss loss function [[bda.local.ml.loss.Loss]]
 * @param minStep minimum step of each iteration, or stop it
 */
case class GBoostStrategy (
    // Required parameters
    // Optional parameters
    var dTreeStrategy: DTreeStrategy = DTreeStrategy.default,
    var numIterations: Int = 100,
    var learningRate: Double = 0.01,
    var loss: Loss = SquaredError,
    var minStep: Double = 1e-5) {
}

object GBoostStrategy {

  def default(): GBoostStrategy = {
    new GBoostStrategy()
  }

}