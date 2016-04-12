package bda.example.cadata

import bda.spark.evaluate.Regression._
import bda.spark.model.tree.gbrt.GBRT
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import bda.spark.reader.Points
import bda.example.{input_dir, tmp_dir}

/**
  * An example app for GBRT(Gradient Boosting Regression Trees) on cadata data set
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
  * The cadata dataset can ben found at `testData/regression/cadata/`.
  * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
  */
object RunSparkGBRT {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark GBRT Training of cadata dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")

    val sc = new SparkContext(conf)
    sc.setCheckpointDir(tmp_dir)

    val data_dir: String = input_dir + "regression/cadata/"

    val train = Points.readLibSVMFile(sc, data_dir + "cadata.train")
    val test = Points.readLibSVMFile(sc, data_dir + "cadata.test")

    train.cache()
    test.cache()

    val gbrt_model = GBRT.train(
      train,
      impurity = "Variance",
      max_depth = 15,
      max_bins = 32,
      bin_samples = 10000,
      min_node_size = 10,
      min_info_gain = 1e-6,
      num_round = 100,
      learn_rate = 0.01)

    // Error of training data set
    val train_preds = gbrt_model.predict(train)
    println(s"Train RMSE: ${RMSE(train_preds)}")

    // Error of testing data set
    val test_preds = gbrt_model.predict(test)
    println(s"Test RMSE: ${RMSE(test_preds)}")
  }
}