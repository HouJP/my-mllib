package mymllib.example.a1a

import mymllib.example._
import mymllib.spark.evaluate.Classification
import mymllib.spark.model.tree.gbdt.GBDT
import mymllib.spark.reader.Points
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}

/**
  * An example app for GBDT(Gradient Boosting Decision Trees) on a1a data set.
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a).
  * The a1a dataset can ben found at `data/classification/a1a`.
  * If you use it as a template to create your own app, please use
  * `spark-submit` to submit your app.
  */
object RunSparkGBDT {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName(s"Spark GBDT Training of a1a dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    val data_dir: String = input_dir + "classification/"
    val train = Points.readLibSVMFile(sc, data_dir + "a1a", is_class = true)
    val test = Points.readLibSVMFile(sc, data_dir + "a1a.t", is_class = true)

    val impurity: String = "Variance"
    val max_depth: Int = 10
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val max_bins: Int = 32
    val bin_samples: Int = 10000
    val num_round: Int = 5

    val gbdt_model = GBDT.train(train,
      impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      num_round)

    // Error of training data set
    val train_preds = gbdt_model.predict(train)
    val train_err = train_preds.filter(r => r._2 != r._3).count().toDouble / train.count()
    println(s"Train Error($train_err)")


    // Error of testing data set
    val test_preds = gbdt_model.predict(test)
    val test_err = test_preds.filter(r => r._2 != r._3).count().toDouble / test.count()
    println(s"Test Error($test_err)")
  }
}