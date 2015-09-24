package bda.local.ml.strategy

import bda.local.ml.impurity.{Variance, Impurity}
import bda.local.ml.loss.{SquaredError, Loss}

/**
 * Class of strategy used for a decision tree.
 *
 * @param impurity information calculator [[bda.local.ml.impurity.Impurity]]
 * @param loss loss function [[bda.local.ml.loss.Loss]]
 * @param minNodeSize minimum size of nodes in the decision tree
 * @param maxDepth maximum depth of nodes in the decision tree
 * @param minInfoGain minimum information gain while splitting, or do not split if not satisfied
 */
class DTreeStrategy(
    var impurity: Impurity = Variance,
    var loss: Loss = SquaredError,
    var minNodeSize: Int = 1,
    var maxDepth: Int = 10,
    var minInfoGain: Double = 0.0) {}

object DTreeStrategy {

  /**
   * Construct a [[DTreeStrategy]] instance with default parameters.
   *
   * @return a [[DTreeStrategy]] instance
   */
  def default: DTreeStrategy = {
    new DTreeStrategy()
  }
}

