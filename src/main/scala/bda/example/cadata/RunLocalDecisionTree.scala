package bda.example.cadata

import scopt.OptionParser
import bda.local.reader.LibSVMFile
import bda.local.model.tree.DecisionTree

/**
 * An example app for DecisionTree on cadata data set in standalone(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
 * The cadata dataset can ben found at `testData/regression/cadata/`.
 */
object RunLocalDecisionTree {
  case class Params(data_dir: String = "/Users/hugh_627/ICT/bda/testData/regression/cadata/",
                    impurity: String = "Variance",
                    loss: String = "SquaredError",
                    max_depth: Int = 10,
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkDecisionTree") {
      head("RunSparkDTree: an example app for DecisionTree on cadata data.")
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
          |   --data_dir hdfs://bda00:8020/user/bda/testData/regression/cadata/
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
      val (data, num) = LibSVMFile.readAsReg(params.data_dir + "cadata.train")
      (data.toArray, num)
    }
    val (valid_data, valid_fs_num) = {
      val (data, num) = LibSVMFile.readAsReg(params.data_dir + "cadata.test")
      (Some(data.toArray), Some(num))
    }

    val dt_model = DecisionTree.train(train_data,
      valid_data,
      params.impurity,
      params.loss,
      params.max_depth,
      params.min_node_size,
      params.min_info_gain)

    val (pred, rmse) = dt_model.predict(valid_data.get)
  }
}