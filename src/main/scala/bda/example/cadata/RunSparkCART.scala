package bda.example.cadata

import bda.spark.reader.Points
import org.apache.spark.{SparkContext, SparkConf}
import bda.spark.model.tree.cart.CART
import bda.example.{input_dir, output_dir}
import org.apache.log4j.{Level, Logger}
import bda.spark.evaluate.Regression.RMSE

/**
  * An example app for DecisionTree on cadata data set.
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
  * The cadata dataset can ben found at `data/regression/cadata/`.
  * If you use it as a template to create your own app, please use
  * `spark-submit` to submit your app.
  */
object RunSparkCART {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val data_dir: String = input_dir + "regression/cadata/"
    val impurity: String = "Variance"
    val max_depth: Int = 10
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val max_bins: Int = 32
    val bin_samples: Int = 10000

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark CART Training of cadata dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")

    val sc = new SparkContext(conf)

    val train = Points.readLibSVMFile(sc, data_dir + "cadata.train")
    val test = Points.readLibSVMFile(sc, data_dir + "cadata.test")

    train.cache()
    test.cache()

    val cart_model = CART.train(
      train,
      impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain)

    cart_model.printStructure()

    // Error of training data set
    val train_preds = cart_model.predict(train)
    println(s"Train RMSE: ${RMSE(train_preds)}")

    // Error of testing data set
    val test_preds = cart_model.predict(test)
    println(s"Test RMSE: ${RMSE(test_preds)}")
  }
}