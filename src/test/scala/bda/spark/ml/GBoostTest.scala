package bda.spark.ml

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.model.LabeledPoint
import bda.spark.ml.para.GBoostPara
import org.apache.spark.SparkContext
import org.scalatest.FunSuite

/**
 * Crate by houjp on 2015-10-13
 */
class GBoostTest extends FunSuite{
  val sc = new SparkContext("local", "MLUtilsTest")

  val lps = Array[LabeledPoint](
    LabeledPoint(1, SparseVector(2, Array[(Int, Double)]((0, 1.0), (1, 2.0)))),
    LabeledPoint(2, SparseVector(2, Array[(Int, Double)]((0, 1.0), (1, 3.0)))))
  val input = sc.parallelize(lps)

  val gb_para = GBoostPara.default

  test("GBoost Test") {
    val gb_model = new GBoost(gb_para).fit(input)

    val pred = gb_model.predict(input)
  }
}