package bda.local.ml

import bda.local.ml.impurity.{Variance, Impurity}
import bda.local.ml.loss.{SquaredError, Loss}

class DTreeStrategy(
    var impurity: Impurity = Variance,
    var loss: Loss = SquaredError,
    var minNodeSize: Int = 1,
    var maxDepth: Int = 10,
    var minInfoGain: Double = 0.0) {}

object DTreeStrategy {

  def default: DTreeStrategy = {
    new DTreeStrategy()
  }
}

