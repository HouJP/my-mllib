package bda.example.dna

import bda.example._
import bda.spark.model.tree.gbdt.GBDT
import bda.spark.model.tree.rf.RandomForest
import bda.spark.reader.Points
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}

/**
  * An example app for Random Forest on dna data set.
  * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#dna).
  * The a1a dataset can ben found at `data/classification/a1a`.
  * If you use it as a template to create your own app, please use
  * `spark-submit` to submit your app.
  */
object RunSparkRandomForest {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark Random Forest Training of dna dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    val data_dir: String = input_dir + "classification/dna/"
    val train = Points.readLibSVMFile(sc, data_dir + "dna.scale")
    val test = Points.readLibSVMFile(sc, data_dir + "dna.scale.t")

    val impurity: String = "Gini"
    val max_depth: Int = 10
    val min_node_size: Int = 20
    val min_info_gain: Double = 1e-6
    val max_bins: Int = 32
    val bin_samples: Int = 10000
    val row_rate: Double = 0.6
    val col_rate: Double = 0.6
    val num_round: Int = 20

    val rf_model = RandomForest.train(train,
      impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_round)

    val train_preds = rf_model.predict(train)
    val train_err = train_preds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println("Train Error = " + train_err)

    val test_preds = rf_model.predict(test)
    val test_err = test_preds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println("Test Error = " + test_err)
  }
}