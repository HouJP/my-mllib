package bda.local.ml.Strategy

import bda.local.ml.loss.{Loss, SquaredError}
import bda.local.ml.loss.SquaredError

case class GBoostStrategy (
    // Required parameters
    // Optional parameters
    var numIterations: Int = 100,
    var learningRate: Double = 0.1,
    var loss: Loss = SquaredError) {
}

object GBoostStrategy {

}