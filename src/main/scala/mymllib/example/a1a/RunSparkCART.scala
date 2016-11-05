package mymllib.example.a1a

import mymllib.example.input_dir
import mymllib.spark.model.tree.cart.CART
import mymllib.spark.reader.Points
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * An example app for CART(Classification And Regression Trees) on a1a data set.
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a).
  * The a1a dataset can ben found at `data/classification/a1a`.
  * If you use it as a template to create your own app, please use
  * `spark-submit` to submit your app.
  */
object RunSparkCART {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val data_dir: String = input_dir + "classification/a1a/"
    val impurity: String = "Gini"
    val max_depth: Int = 10
    val min_node_size: Int = 50
    val min_info_gain: Double = 1e-6
    val max_bins: Int = 32
    val bin_samples: Int = 10000

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark CART Training of a1a dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")

    val sc = new SparkContext(conf)

    val train = Points.readLibSVMFile(sc, data_dir + "a1a", true)
    val test = Points.readLibSVMFile(sc, data_dir + "a1a.t", true)

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

    val preds = cart_model.predict(test)

    val err = preds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println("Test Error = " + err)
  }
}