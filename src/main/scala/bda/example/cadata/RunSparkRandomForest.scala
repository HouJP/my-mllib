package bda.example.cadata

import bda.spark.evaluate.Regression._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import bda.spark.model.tree.rf.{RandomForestModel, RandomForest}
import bda.spark.reader.Points
import bda.example.{input_dir, tmp_dir}

/**
 * An example app for random forest on cadata data set
 * (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
 * The cadata dataset can ben found at `testData/regression/cadata/`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RunSparkRandomForest {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val data_dir: String = input_dir + "regression/cadata/"
    val impurity: String = "Variance"
    val max_depth: Int = 10
    val max_bins: Int = 32
    val min_samples: Int = 10000
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val row_rate = 0.6
    val col_rate = 0.6
    val num_trees = 20

    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"Spark Random Forest Training of cadata dataset")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(tmp_dir)

    val train = Points.readLibSVMFile(sc, data_dir + "cadata.train", false)
    val test = Points.readLibSVMFile(sc, data_dir + "cadata.test", false)

    train.cache()
    test.cache()

    val rf_model: RandomForestModel = RandomForest.train(
      train,
      impurity,
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees)

    // Error of training data set
    val train_preds = rf_model.predict(train).map(e => (e._2, e._3))
    println(s"Train RMSE: ${RMSE(train_preds)}")

    // Error of testing data set
    val test_preds = rf_model.predict(test).map(e => (e._2, e._3))
    println(s"Test RMSE: ${RMSE(test_preds)}")
  }
}