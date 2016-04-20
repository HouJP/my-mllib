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

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark CART Training of cadata dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")

    val sc = new SparkContext(conf)

    val data_dir: String = input_dir + "regression/"
    val data = Points.readLibSVMFile(sc, data_dir + "/cadata", is_class = false)
    val Array(train, test) = data.randomSplit(Array(0.75, 0.25))

    train.cache()
    test.cache()

    val cart_model = CART.train(
      train,
      impurity = "Variance",
      max_depth = 15,
      max_bins = 32,
      bin_samples = 10000,
      min_node_size = 10,
      min_info_gain = 1e-6)

    cart_model.printStructure()

    // Error of training data set
    val train_preds = cart_model.predict(train).map(e => (e._2, e._3))
    println(s"Train RMSE: ${RMSE(train_preds)}")

    // Error of testing data set
    val test_preds = cart_model.predict(test).map(e => (e._2, e._3))
    println(s"Test RMSE: ${RMSE(test_preds)}")
  }
}