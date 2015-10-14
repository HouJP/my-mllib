package bda.spark.ml.util

import org.apache.spark.SparkContext
import org.scalatest.FunSuite

/**
 * Created by houjp on 2015-10-09
 */
class MLUtilsTest extends FunSuite {
  val sc = new SparkContext("local", "MLUtilsTest")
  val pt = "/Users/hugh_627/ICT/bda/gboost/data/mydata"

  test("MLUtils Test") {
    val rdd = MLUtils.loadLibSVMFile(sc, pt)

    val f_sz = 2 // feature size
    val d_sz = 4 // number of data

    assert(rdd.count() == d_sz)
    assert(rdd.take(1)(0).features.size == f_sz)

    // show data
    for (i <- 0 until d_sz) {
      println(s"Point $i: ${rdd.take(i + 1)(i)}")
    }
  }
}