package bda.examples.ml

import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser

import bda.spark.ml.util.MLUtils
import bda.spark.ml.para.{Loss, Impurity, DTreePara}
import bda.spark.ml.DTree

object RunSparkDTree {
  case class Params (
      input: String = null,
      loss: String = "SquaredError",
      min_step: Double = 1e-5,
      impurity: String = "Variance",
      min_node_size: Int = 15,
      max_depth: Int = 10,
      max_bins: Int = 32,
      min_samples: Int = 10000)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkDTree") {
      head("RunSparkDTree: an example app for GBTs on YearPredictionMSD data.")
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
        .text(s"maximum bins's number of tree, default: ${default_params.max_bins}")
        .action((x, c) => c.copy(max_bins = x))
      opt[Int]("min_samples")
        .text(s"minimum number of samples of tree, default: ${default_params.min_samples}")
        .action((x, c) => c.copy(min_samples = x))
      arg[String]("<input>")
        .required()
        .text("input paths to the cadata dataset in LibSVM format")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on the YearPredictionMSD dataset:
          |
          | bin/spark-submit --class bda.examples.ml.RunSparkDTree \
          |   out/artifacts/*/*.jar \
          |   --min_node_size 15 --max_depth 10 \
          |   --max_bins 32 --min_samples 10000 \
          |   hdfs://bda00:8020/user/houjp/gboost/data/YearPredictionMSD/
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"DTree Example with $params")
    val sc = new SparkContext(conf)

    // Load and parse the data file
    val train_data = MLUtils.loadLibSVMFile(sc, params.input + ".train")
    val test_data = MLUtils.loadLibSVMFile(sc, params.input + ".test")

    // Set DTree parameters
    val dt_para = new DTreePara(
      impurity = Impurity.fromString(params.impurity),
      loss = Loss.fromString(params.loss),
      min_node_size = params.min_node_size,
      max_depth = params.max_depth,
      max_bins = params.max_bins,
      min_samples = params.min_samples)

    // Train a DTree model.
    val model = new DTree(dt_para).fit(train_data)

    // Evaluate model on train instances and compute train RMSE
    model.predict(train_data)

    // Evaluate model on test instances and compute test RMSE
    model.predict(test_data)
  }
}