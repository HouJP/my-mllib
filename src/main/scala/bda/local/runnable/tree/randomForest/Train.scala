package bda.local.runnable.tree.randomForest

import bda.local.reader.Points.readLibSVMFile
import scopt.OptionParser
import bda.local.model.tree.{RandomForestModel, RandomForest}

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
                    min_samples: Int = 10000,
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6,
                    row_rate: Double = 0.6,
                    col_rate: Double = 0.6,
                    num_trees: Int = 20,
                    feature_num: Int = 0)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunLocalRandomForest") {
      head("RunLocalRandomForest: an example app for RandomForest.")
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
      opt[Int]("min_samples")
        .text(s"minimum number of samples of tree, default: ${default_params.min_samples}")
        .action((x, c) => c.copy(min_samples = x))
      opt[Int]("min_node_size")
        .text(s"minimum node size, default: ${default_params.min_node_size}")
        .action((x, c) => c.copy(min_node_size = x))
      opt[Double]("min_info_gain")
        .text(s"minimum information gain, default: ${default_params.min_info_gain}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Double]("row_rate")
        .text(s"sample ratio of training data set, default: ${default_params.row_rate}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Double]("col_rate")
        .text(s"sample ratio of features, default: ${default_params.col_rate}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Int]("num_trees")
        .text(s"number of decision trees, default: ${default_params.num_trees}")
        .action((x, c) => c.copy(num_trees = x))
      opt[Int]("feature_num").required()
        .text(s"number of features, default: ${default_params.feature_num}")
        .action((x, c) => c.copy(feature_num = x))
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
          | java -jar out/artifacts/*/*.jar \
          |   --impurity "Variance" --loss "SquaredError" \
          |   --max_depth 10 --max_bins 32 \
          |   --min_samples 10000 --min_node_size 15 \
          |   --min_info_gain 1e-6 --num_trees 20\
          |   --feature_num 8 \
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
    val points = readLibSVMFile(params.train_pt).toSeq

    // prepare training and validate datasets
    val (train, test) = if (!params.valid_pt.isEmpty) {
      val points2 = readLibSVMFile(params.valid_pt).toSeq
      (points, points2)
    } else {
      // train without validation
      (points, null)
    }

    val model: RandomForestModel = RandomForest.train(train,
      test,
      params.feature_num,
      params.impurity,
      params.loss,
      params.max_depth,
      params.max_bins,
      params.min_samples,
      params.min_node_size,
      params.min_info_gain,
      params.row_rate,
      params.col_rate,
      params.num_trees)

    if (!params.model_pt.isEmpty)
      model.save(params.model_pt)
  }
}