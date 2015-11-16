package bda.spark.ml.para

import bda.spark.ml.para.Impurity.Variance
import bda.spark.ml.para.Impurity._
import bda.spark.ml.impurity.VarianceCalculator
import bda.spark.ml.loss.SquaredErrorCalculator
import bda.spark.ml.para.Loss.{Loss, SquaredError}

class DTreePara(
    val impurity: Impurity = Variance,
    val loss: Loss = SquaredError,
    val max_depth: Int = 10,
    val max_bins: Int = 32,
    val min_samples: Int = 10000,
    val min_node_size: Int = 15) extends  Serializable {

  val impurity_calculator = impurity match {
    case Variance => VarianceCalculator
    case _ => throw new IllegalAccessException(s"Did not recognize impurity type: $impurity")
  }

  val loss_calculator = loss match {
    case SquaredError => SquaredErrorCalculator
    case _ => throw new IllegalArgumentException(s"Did not recognize loss type: $loss")
  }
}

object DTreePara {

  def default: DTreePara = {
    new DTreePara()
  }

  def defaultPara(impurity: String, loss: String): DTreePara = {
    new DTreePara(Impurity.fromString(impurity), Loss.fromString(loss))
  }

  def defaultPara(impurity: Impurity, loss: Loss): DTreePara = {
    new DTreePara(impurity, loss)
  }
}