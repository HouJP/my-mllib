package bda.spark.runnable.tree.randomForest

import bda.common.obj.LabeledPoint
import bda.spark.reader.Points
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import bda.spark.model.tree.{RandomForestModel, RandomForest}
import org.apache.log4j.{Level, Logger}

/**
 * Command line runner for spark random forest.
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
                    impurity: String = "Variance",
                    loss: String = "SquaredError",
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

    val parser = new OptionParser[Params]("RunSparkRandomForest") {
      head("RunSparkRandomForest: an example app for Random Forest.")
      opt[String]("impurity")
        .text(s"impurity of each node, default: ${default_params.impurity}")
        .action((x, c) => c.copy(impurity = x))
      opt[String]("loss")
        .text(s"loss function, default: ${default_params.loss}")
        .action((x, c) => c.copy(loss = x))
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
      opt[Double]("row_rate")
        .text(s"sampling rate of training data set, default: ${default_params.row_rate}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Double]("col_rate")
        .text(s"sampling rate of features, default: ${default_params.col_rate}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Int]("num_trees")
        .text(s"number of decision trees, default: ${default_params.num_trees}")
        .action((x, c) => c.copy(num_trees = x))
      opt[String]("train_pt").required()
        .text("input paths to the training dataset in LibSVM format")
        .action((x, c) => c.copy(train_pt = x))
      opt[String]("valid_pt")
        .text("input paths to the validation dataset in LibSVM format")
        .action((x, c) => c.copy(valid_pt = x))
      opt[String]("model_pt")
        .text("directory of the decision tree model")
        .action((x, c) => c.copy(model_pt = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | bin/spark-submit --class bda.runnable.tree.randomForest.Train \
          |   out/artifacts/*/*.jar \
          |   --impurity "Variance" --loss "SquaredError" \
          |   --max_depth 10 --max_bins 32 \
          |   --min_samples 10000 --min_node_size 15 \
          |   --min_info_gain 1e-6 --row_rate 0.6 \
          |   --col_rate 0.6 --num_trees 50\
          |   --train_pt ... \
          |   --valid_pt ... \
          |   --model_pt ...
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"Spark Random Forest Training")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    val points = sc.textFile(params.train_pt).map(LabeledPoint.parse)

    // prepare training and validate datasets
    val (train_points, valid_points) = if (!params.valid_pt.isEmpty) {
      val points2 = sc.textFile(params.valid_pt).map(LabeledPoint.parse)
      (points, points2)
    } else {
      // train without validation
      (points, null)
    }

    train_points.cache()
    if (valid_points != null) valid_points.cache()

    val model: RandomForestModel = RandomForest.train(
      train_points,
      valid_points,
      params.impurity,
      params.loss,
      params.max_depth,
      params.max_bins,
      params.bin_samples,
      params.min_node_size,
      params.min_info_gain,
      params.row_rate,
      params.col_rate,
      params.num_trees)

    if (!params.model_pt.isEmpty) {
      model.save(sc, params.model_pt)
    }
  }
}