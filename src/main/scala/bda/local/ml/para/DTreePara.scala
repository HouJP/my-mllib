package bda.local.ml.para

import bda.local.ml.impurity.{Variance, Impurity}
import bda.local.ml.loss.{LossCounter, SquaredErrorCalculator, LossCalculator}
import bda.local.ml.para.Loss.{Loss, SquaredError}

import scala.beans.BeanProperty

/**
 * Class of strategy used for a decision tree.
 *
 * @param impurity information calculator [[bda.local.ml.impurity.Impurity]]
 * @param loss loss type [[bda.local.ml.para.Loss]]
 * @param min_node_size minimum size of nodes in the decision tree
 * @param max_depth maximum depth of nodes in the decision tree
 * @param min_info_gain minimum information gain while splitting, or do not split if not satisfied
 */
class DTreePara(
    val impurity: Impurity = Variance,
    val loss: Loss = Loss.fromString("SquaredError"),
    val min_node_size: Int = 15,
    val max_depth: Int = 10,
    val min_info_gain: Double = 0.0) {

  val loss_calculator = loss match {
    case SquaredError => SquaredErrorCalculator
    case _ => throw new IllegalArgumentException(s"Did not recognize loss type $SquaredError")
  }
}

object DTreePara {

  /**
   * Construct a [[DTreePara]] instance with default parameters.
   *
   * @return a [[DTreePara]] instance
   */
  def default: DTreePara = {
    new DTreePara()
  }
}

