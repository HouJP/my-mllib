package bda.example.cadata

import bda.local.preprocess.Points
import scopt.OptionParser
import bda.local.model.tree.{DecisionTree, DecisionTreeModel}

/**
 * An example app for DecisionTree on cadata data set in standalone(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
 * The cadata dataset can ben found at `testData/regression/cadata/`.
 */
object RunLocalDecisionTree {
  case class Params(data_dir: String = "/Users/hugh_627/ICT/bda/testData/regression/cadata/",
                    feature_num: Int = 8,
                    impurity: String = "Variance",
                    loss: String = "SquaredError",
                    max_depth: Int = 10,
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkDecisionTree") {
      head("RunSparkDecisionTree: an example app for DecisionTree on cadata data.")
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
      opt[Int]("feature_num")
        .text(s"number of features, default: ${default_params.feature_num}")
        .action((x, c) => c.copy(min_node_size = x))
      opt[String]("data_dir")
        .text("path to the cadata dataset in LibSVM format")
        .action((x, c) => c.copy(data_dir = x))
      note(
        """
          |For example, the following command runs this app on the cadata dataset:
          |
          | java -jar out/artifacts/*/*.jar \
          |   --impurity "Variance" --loss "SquaredError" \
          |   --max_depth 10 --max_bins 32 \
          |   --min_samples 10000 --min_node_size 15 \
          |   --min_info_gain 1e-6 \
          |   --feature_num 8 \
          |   --data_dir ...
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {

    val train = Points.fromLibSVMFile(params.data_dir + "/cadata.train", params.feature_num).toSeq
    val test = Points.fromLibSVMFile(params.data_dir + "/cadata.test", params.feature_num).toSeq

    val model: DecisionTreeModel = DecisionTree.train(train,
      test,
      params.feature_num,
      params.impurity,
      params.loss,
      params.max_depth,
      params.min_node_size,
      params.min_info_gain)
  }
}