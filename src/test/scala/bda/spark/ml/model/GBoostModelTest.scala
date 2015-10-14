package bda.spark.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.DTree
import bda.local.ml.loss.SquaredErrorCalculator
import bda.local.ml.model.LabeledPoint
import bda.local.ml.para.DTreePara
import org.apache.spark.SparkContext
import org.scalatest.FunSuite

/**
 * Crate by houjp on 2015-10-13
 */
class GBoostModelTest extends FunSuite {
  val sc = new SparkContext("local", "MLUtilsTest")

  val lps = Array[LabeledPoint](
    LabeledPoint(1, SparseVector(2, Array[(Int, Double)]((0, 1.0), (1, 2.0)))),
    LabeledPoint(2, SparseVector(2, Array[(Int, Double)]((0, 1.0), (1, 3.0)))))

  val dt_para = DTreePara.default
  val dt_model = new DTree(dt_para).fit(lps)

  val input = sc.parallelize(lps)
  val loss = SquaredErrorCalculator
  val weight = 0.5

  test("GBoostModel Test") {
    val pred_err = GBoostModel.computePredictAndError(input, weight, dt_model, loss)
    pred_err.foreach(println)

    val new_pred_err = GBoostModel.updatePredictAndError(input, pred_err, weight, dt_model, loss)
    new_pred_err.foreach(println)
  }
}