package bda.spark.runnable.tree.gbrt

import bda.common.obj.LabeledPoint
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import bda.spark.model.tree.gbrt.GBRT

/**
  * Command line runner for spark GBRT(Gradient Boosting Regression Trees).
  *
  * Input:
  * - train_pt format: label fid1:v1 fid2:v2 ...
  * Both label and v are doubles, fid are integers starting from 1.
  */
object Train {

  /** command line parameters */
  case class Params(train_pt: String = "",
                    valid_pt: String = "",
                    model_pt: String = "",
                    cp_pt: String = "",
                    impurity: String = "Variance",
                    max_depth: Int = 10,
                    max_bins: Int = 32,
                    bin_samples: Int = 10000,
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6,
                    num_round: Int = 10,
                    learn_rate: Double = 0.02)

  def main(args: Array[String]) {
    // do not show log info
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkGBRTTrain") {
      head("RunSparkGBRTTrain: an example app for GBRT.")
      opt[String]("train_pt").required()
        .text("input paths to the training dataset in LibSVM format")
        .action((x, c) => c.copy(train_pt = x))
      opt[String]("valid_pt")
        .text("input paths to the validation dataset in LibSVM format")
        .action((x, c) => c.copy(valid_pt = x))
      opt[String]("model_pt").required()
        .text("directory of the GBRT model")
        .action((x, c) => c.copy(model_pt = x))
      opt[String]("cp_pt").required()
        .text("checkpoint directory")
        .action((x, c) => c.copy(cp_pt = x))
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
        .text(s"minimum information gain, default: ${default_params.min_info_gain}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Int]("num_round")
        .text(s"number of round, default: ${default_params.num_round}")
        .action((x, c) => c.copy(num_round = x))
      opt[Double]("learn_rate")
        .text(s"learning rate, default: ${default_params.learn_rate}")
        .action((x, c) => c.copy(learn_rate = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | bin/spark-submit --class bda.runnable.tree.gbrt.GBRT \
          |   out/artifacts/*/*.jar \
          |   --impurity "Variance" \
          |   --max_depth 10 \
          |   --max_bins 32 \
          |   --min_samples 10000 \
          |   --min_node_size 15 \
          |   --min_info_gain 1e-6 \
          |   --num_round 10 \
          |   --learn_rate 0.02 \
          |   --train_pt ... \
          |   --valid_pt ... \
          |   --model_pt ... \
          |   --cp_pt ...
        """.stripMargin)
    }

    parser.parse(args, default_params) match {
      case Some(params) => run(params)
      case None => System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"Spark GBRT Training")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(params.cp_pt)

    // prepare training
    val train = sc.textFile(params.train_pt).map(LabeledPoint.parse).cache()
    val watchlist = if (!params.valid_pt.isEmpty) {
      Array(("valid", sc.textFile(params.valid_pt).map(LabeledPoint.parse).cache()))
    } else {
      Array[(String, RDD[LabeledPoint])]()
    }

    val model = GBRT.train(
      train,
      watchlist,
      params.impurity,
      params.max_depth,
      params.max_bins,
      params.bin_samples,
      params.min_node_size,
      params.min_info_gain,
      params.num_round,
      params.learn_rate)

    model.save(sc, params.model_pt)
  }
}