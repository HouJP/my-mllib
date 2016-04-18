package bda.spark.runnable.tree.rf

import bda.common.obj.LabeledPoint
import bda.spark.model.tree.rf.RandomForest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser

/**
  * Command line runner for spark Random Forest.
  *
  * Input:
  * - train_pt format: label fid1:v1 fid2:v2 ...
  * Both label and v are doubles, fid are integers starting from 1.
  */
object Train {

  /** command line parameters */
  case class Params(train_pt: String = "",
                    model_pt: String = "",
                    impurity: String = "Variance",
                    max_depth: Int = 10,
                    max_bins: Int = 32,
                    bin_samples: Int = 10000,
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6,
                    row_rate: Double = 0.6,
                    col_rate: Double = 0.6,
                    num_trees: Int = 20)

  def main(args: Array[String]) {
    // do not show log info
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkRFTrain") {
      head("RunSparkRFTrain: an example app for Random Forest.")
      opt[String]("train_pt").required()
        .text("input paths to the training dataset in LibSVM format")
        .action((x, c) => c.copy(train_pt = x))
      opt[String]("model_pt").required()
        .text("directory of the Random Forest model")
        .action((x, c) => c.copy(model_pt = x))
      opt[String]("impurity")
        .text(s"impurity of each node, default: ${default_params.impurity}")
        .action((x, c) => c.copy(impurity = x))
      opt[Int]("max_depth")
        .text(s"maximum depth of tree, default: ${default_params.max_depth}")
        .action((x, c) => c.copy(max_depth = x))
      opt[Int]("max_bins")
        .text(s"maximum bins's number of tree, default: ${default_params.max_bins}")
        .action((x, c) => c.copy(max_bins = x))
      opt[Int]("bin_samples")
        .text(s"minimum number of samples of tree, default: ${default_params.bin_samples}")
        .action((x, c) => c.copy(bin_samples = x))
      opt[Int]("min_node_size")
        .text(s"minimum node size, default: ${default_params.min_node_size}")
        .action((x, c) => c.copy(min_node_size = x))
      opt[Double]("min_info_gain")
        .text(s"minimum information gaint: ${default_params.min_info_gain}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Double]("row_rate")
        .text(s"sample ratio of train data set: ${default_params.row_rate}")
        .action((x, c) => c.copy(row_rate = x))
      opt[Double]("col_rate")
        .text(s"sample ratio of features: ${default_params.col_rate}")
        .action((x, c) => c.copy(col_rate = x))
      opt[Int]("num_trees")
        .text(s"number of decision trees, default: ${default_params.num_trees}")
        .action((x, c) => c.copy(num_trees = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | bin/spark-submit --class bda.runnable.tree.cart.Train \
          |   out/artifacts/*/*.jar \
          |   --impurity "Variance" \
          |   --max_depth 10 \
          |   --max_bins 32 \
          |   --min_samples 10000 \
          |   --min_node_size 15 \
          |   --min_info_gain 1e-6 \
          |   --row_rate 0.6 \
          |   --col_rate 0.6 \
          |   --num_trees 20 \
          |   --train_pt ... \
          |   --model_pt ...
        """.stripMargin)
    }

    parser.parse(args, default_params) match {
      case Some(params) => run(params)
      case None => System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"Spark Random Forest Training")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    // prepare training
    val train = sc.textFile(params.train_pt).map(LabeledPoint.parse).cache()

    val model = RandomForest.train(
      train,
      params.impurity,
      params.max_depth,
      params.max_bins,
      params.bin_samples,
      params.min_node_size,
      params.min_info_gain,
      params.row_rate,
      params.col_rate,
      params.num_trees)

    model.save(sc, params.model_pt)
  }
}