package bda.local.ml.Strategy

import bda.local.ml.loss.{Loss, SquaredError}

/**
 * Class of strategy for GBoost.
 *
 * @param numIterations number of iterations
 * @param learningRate vlaue of learning rate
 * @param loss loss function [[bda.local.ml.loss.Loss]]
 */
case class GBoostStrategy (
    // Required parameters
    // Optional parameters
    var numIterations: Int = 100,
    var learningRate: Double = 0.1,
    var loss: Loss = SquaredError) {
}

object GBoostStrategy {

}