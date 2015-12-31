package bda.example.cadata

import org.apache.spark.{SparkContext, SparkConf}
import bda.spark.model.tree.{GradientBoostModel, GradientBoost}
import bda.spark.reader.Points
import bda.example.{input_dir, tmp_dir}

/**
  * An example app for GradientBoost on cadata data set
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
  * The cadata dataset can ben found at `testData/regression/cadata/`.
  * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
  */
object RunSparkGradientBoost {

  def main(args: Array[String]) {
    val data_dir: String = input_dir + "regression/cadata/"
    val feature_num: Int = 8
    val impurity: String = "Variance"
    val loss: String = "SquaredError"
    val max_depth: Int = 10
    val max_bins: Int = 32
    val min_samples: Int = 10000
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val row_rate = 0.6
    val col_rate = 0.6
    val num_iter: Int = 50
    val learn_rate: Double = 0.02
    val min_step: Double = 1e-5

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark Gradient Boost Training of cadata dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")

    val sc = new SparkContext(conf)
    sc.setCheckpointDir(tmp_dir)

    val train = Points.readLibSVMFile(sc, data_dir + "cadata.train")
    val test = Points.readLibSVMFile(sc, data_dir + "cadata.test")

    train.cache()
    test.cache()

    val model: GradientBoostModel = GradientBoost.train(
      train,
      test,
      feature_num,
      impurity,
      loss,
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_iter,
      learn_rate,
      min_step)
  }
}