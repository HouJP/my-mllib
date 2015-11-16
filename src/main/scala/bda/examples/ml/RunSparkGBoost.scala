package bda.examples.ml

import bda.spark.ml.para.{Loss, Impurity, DTreePara}
import scopt.OptionParser
import bda.spark.ml.para.GBoostPara
import bda.spark.ml.util.MLUtils
import bda.spark.ml.GBoost
import bda.spark.ml.model.GBoostModel
import org.apache.spark.{SparkConf, SparkContext}
import bda.local.ml.util.Log

/**
 * An example app for GBoost on YearPredictionMSD data (http://archive.ics.uci.edu/ml/YearPredictionMSD).
 * The YearPredictionMSD dataset can ben found at `data/YearPredictionMSD`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RunSparkGBoost {

  case class Params (
      input: String = null,
      cp_dir: String = null,
      num_iter: Int = 200,
      learn_rate: Double = 0.2,
      loss: String = "SquaredError",
      min_step: Double = 1e-5,
      impurity: String = "Variance",
      min_node_size: Int = 15,
      max_depth: Int = 10,
      max_bins: Int = 32,
      min_samples: Int = 10000)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkGBoost") {
      head("RunSparkGBoost: an example app for GBoost on YearPredictionMSD data.")
      opt[Int]("num_iter")
        .text(s"number of iterations, default: ${default_params.num_iter}")
        .action((x, c) => c.copy(num_iter = x))
      opt[Double]("learn_rate")
        .text(s"learning rate, default: ${default_params.learn_rate}")
        .action((x, c) => c.copy(learn_rate = x))
      opt[String]("loss")
        .text(s"loss function, default: ${default_params.loss}")
        .action((x, c) => c.copy(loss = x))
      opt[Double]("min_step")
        .text(s"min step of each iteration, default: ${default_params.min_step}")
        .action((x, c) => c.copy(min_step = x))
      opt[String]("impurity")
        .text(s"impurity of each node, default: ${default_params.impurity}")
        .action((x, c) => c.copy(impurity = x))
      opt[Int]("min_node_size")
        .text(s"minimum node size, default: ${default_params.min_node_size}")
        .action((x, c) => c.copy(min_node_size = x))
      opt[Int]("max_depth")
        .text(s"maximum depth of tree, default: ${default_params.max_depth}")
        .action((x, c) => c.copy(max_depth = x))
      opt[Int]("max_bins")
        .text(s"maximum number of bins, default: ${default_params.max_bins}")
        .action((x, c) => c.copy(max_bins = x))
      opt[Int]("min_samples")
        .text(s"minimum number of samples, default: ${default_params.min_samples}")
        .action((x, c) => c.copy(min_samples = x))
      arg[String]("<input>")
        .required()
        .text("input paths to the cadata dataset in LibSVM format")
        .action((x, c) => c.copy(input = x))
      arg[String]("<cp_dir>")
        .required()
        .text("checkpoint directory")
        .action((x, c) => c.copy(cp_dir = x))
      note(
        """
          |For example, the following command runs this app on the cadata dataset:
          |
          | bin/spark-submit --class bda.examples.ml.RunSparkGBoost \
          |   out/artifacts/*/*.jar \
          |   --num_iter 100 --learn_rate 0.01 \
          |   --min_node_size 15 --max_depth 10 \
          |   --max_bins 32 --min_samples 10000 \
          |   hdfs://bda00:8020/user/houjp/gboost/data/YearPredictionMSD/
          |   hdfs://bda00:8020/user/houjp/gboost/data/checkpoint/
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"GBoost Example with $params")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(params.cp_dir)

    val dt_para = new DTreePara(
      impurity = Impurity.fromString(params.impurity),
      loss = Loss.fromString(params.loss),
      min_node_size = params.min_node_size,
      max_depth = params.max_depth,
      max_bins = params.max_bins,
      min_samples = params.min_samples)
    val gb_para = new GBoostPara(dt_para = dt_para,
      num_iter = params.num_iter,
      learn_rate = params.learn_rate,
      loss = Loss.fromString(params.loss),
      min_step = params.min_step)

    // Load and parse the data file
    val train_data = MLUtils.loadLibSVMFile(sc, params.input + ".train").persist()
    val test_data = MLUtils.loadLibSVMFile(sc, params.input + ".test").persist()

    val gb_model: GBoostModel = new GBoost(gb_para).fit(train_data)

    Log.log("INFO", "get train set RMSE")
    gb_model.predict(train_data)
    Log.log("INFO", "get test set RMSE")
    gb_model.predict(test_data)

    train_data.unpersist()
    test_data.unpersist()
  }
}