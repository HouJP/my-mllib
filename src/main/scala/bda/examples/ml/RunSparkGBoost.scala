package bda.examples.ml

import bda.spark.ml.para.GBoostPara
import bda.spark.ml.util.MLUtils
import bda.spark.ml.GBoost
import org.apache.spark.{SparkConf, SparkContext}
import bda.local.ml.util.Log

object RunSparkGBoost {

  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("GBoost Example on Spark"))

    val train_pt = "hdfs://bda00:8020/user/houjp/gboost/data/cadata/cadata.part1"
    val test_pt = "hdfs://bda00:8020/user/houjp/gboost/data/cadata/cadata.part2"

    val train = MLUtils.loadLibSVMFile(sc, train_pt)
    val test = MLUtils.loadLibSVMFile(sc, test_pt)

    val gb_para = GBoostPara.default

    val gb_model = new GBoost(gb_para).fit(train)

    Log.log("INFO", "get train set RMSE")
    gb_model.predict(train)
    Log.log("INFO", "get test set RMSE")
    gb_model.predict(test)
  }
}