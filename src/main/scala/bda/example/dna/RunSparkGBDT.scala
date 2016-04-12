package bda.example.dna

import bda.example._
import bda.spark.model.tree.gbdt.GBDT
import bda.spark.reader.Points
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}

/**
  * An example app for GBDT(Gradient Boosting Decision Tree) on dna data set.
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#dna).
  * The a1a dataset can ben found at `data/classification/a1a`.
  * If you use it as a template to create your own app, please use
  * `spark-submit` to submit your app.
  */
object RunSparkGBDT {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark GBDT Training of dna dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    val data_dir: String = input_dir + "classification/dna/"
    val train = Points.readLibSVMFile(sc, data_dir + "dna.scale")
    val test = Points.readLibSVMFile(sc, data_dir + "dna.scale.t")

    val impurity: String = "Variance"
    val max_depth: Int = 10
    val min_node_size: Int = 20
    val min_info_gain: Double = 1e-6
    val max_bins: Int = 32
    val bin_samples: Int = 10000
    val num_round: Int = 3

    val gbdt_model = GBDT.train(train,
      impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      num_round)

    val train_preds = gbdt_model.predict(train)
    val train_err = train_preds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println("Train Error = " + train_err)

    val test_preds = gbdt_model.predict(test)
    val test_err = test_preds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println("Test Error = " + test_err)
  }
}