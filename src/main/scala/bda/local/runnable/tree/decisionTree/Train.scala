package bda.local.runnable.decisionTree

import scopt.OptionParser
import bda.local.reader.LibSVMFile
import bda.local.model.tree.DecisionTree

/**
 * Command line runner for local decision tree.
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
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("bda.local.runnable.decisionTree.Train") {
      head("Train: an example app for standalone DecisionTree.")
      opt[String]("impurity")
        .text(s"impurity of each node, default: ${default_params.impurity}")
        .action((x, c) => c.copy(impurity = x))
      opt[String]("loss")
        .text(s"loss function, default: ${default_params.loss}")
        .action((x, c) => c.copy(loss = x))
      opt[Int]("max_depth")
        .text(s"maximum depth of tree, default: ${default_params.max_depth}")
        .action((x, c) => c.copy(max_depth = x))
      opt[Int]("min_node_size")
        .text(s"minimum node size, default: ${default_params.min_node_size}")
        .action((x, c) => c.copy(min_node_size = x))
      opt[Double]("min_info_gain")
        .text(s"minimum information gain, default: ${default_params.min_info_gain}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[String]("train_pt")
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
          |   --min_info_gain 1e-6 \
          |   --train_pt /user/houjp/data/YourTrainingData/
          |   --valid_pt /user/houjp/data/YourValidationData/
          |   --model_pt /user/houjp/model/YourModelName/
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {

    // Load and parse the data file
    val (train_data, train_fs_num) = {
      val (data, num) = LibSVMFile.readAsReg(params.train_pt)
      (data.toArray, num)
    }
    val (valid_data, valid_fs_num) = params.valid_pt.isEmpty match {
      case true => (None, 0)
      case false => {
        val (data, num) = LibSVMFile.readAsReg(params.valid_pt)
        (Some(data.toArray), Some(num))
      }
    }

    val dt_model = DecisionTree.train(train_data,
      valid_data,
      params.impurity,
      params.loss,
      params.max_depth,
      params.min_node_size,
      params.min_info_gain)

    if (!params.model_pt.isEmpty) {
      dt_model.save(params.model_pt)
    }
  }
}