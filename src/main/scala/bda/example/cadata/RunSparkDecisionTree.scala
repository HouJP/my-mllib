package bda.example.cadata

import bda.spark.preprocess.Points
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import bda.spark.model.tree.{DecisionTreeModel, DecisionTree}

/**
 * An example app for DecisionTree on cadata data set(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
 * The cadata dataset can ben found at `testData/regression/cadata/`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RunSparkDecisionTree {
  case class Params(data_dir: String = "/Users/hugh_627/ICT/bda/testData/regression/cadata/",
                    feature_num: Int = 8,
                    impurity: String = "Variance",
                    loss: String = "SquaredError",
                    max_depth: Int = 10,
                    max_bins: Int = 32,
                    min_samples: Int = 10000,
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
      opt[Int]("feature_num")
        .text(s"number of features, default: ${default_params.feature_num}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[String]("data_dir")
        .text("path to the cadata dataset in LibSVM format")
        .action((x, c) => c.copy(data_dir = x))
      note(
        """
          |For example, the following command runs this app on the cadata dataset:
          |
          | bin/spark-submit --class bda.example.tree.RunSparkDecisionTree \
          |   out/artifacts/*/*.jar \
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
    val conf = new SparkConf().setAppName(s"Spark Decision Tree Training of cadata dataset")
    val sc = new SparkContext(conf)

    val train = Points.fromLibSVMFile(sc, params.data_dir + "/cadata.train", params.feature_num)
    val test = Points.fromLibSVMFile(sc, params.data_dir + "/cadata.test", params.feature_num)

    train.cache()
    test.cache()

    val model: DecisionTreeModel = DecisionTree.train(
      train,
      test,
      params.feature_num,
      params.impurity,
      params.loss,
      params.max_depth,
      params.max_bins,
      params.min_samples,
      params.min_node_size,
      params.min_info_gain)
  }
}