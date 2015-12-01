package bda.spark.runnable.gradientBoost

import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import bda.spark.reader.LibSVMFile
import bda.spark.model.tree.GradientBoost

/**
 * Command line runner for spark gradient boost.
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
                    cp_dir: String = "",
                    impurity: String = "Variance",
                    loss: String = "SquaredError",
                    max_depth: Int = 10,
                    max_bins: Int = 32,
                    min_samples: Int = 10000,
                    min_node_size: Int = 15,
                    min_info_gain: Double = 1e-6,
                    num_iter: Int = 50,
                    learn_rate: Double = 0.02,
                    min_step: Double = 1e-5)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkGradientBoost") {
      head("RunSparkGradientBoost: an example app for GradientBoost.")
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
      opt[Int]("num_iter")
        .text(s"number of iterations, default: ${default_params.num_iter}")
        .action((x, c) => c.copy(num_iter = x))
      opt[Double]("learn_rate")
        .text(s"Learning rate, default: ${default_params.learn_rate}")
        .action((x, c) => c.copy(learn_rate = x))
      opt[Double]("min_step")
        .text(s"minimum step, default: ${default_params.min_step}")
        .action((x, c) => c.copy(min_step = x))
      opt[String]("train_pt")
        .text("input paths to the training dataset in LibSVM format")
        .action((x, c) => c.copy(train_pt = x))
      opt[String]("valid_pt")
        .text("input paths to the validation dataset in LibSVM format")
        .action((x, c) => c.copy(valid_pt = x))
      opt[String]("model_pt")
        .text("directory of the decision tree model")
        .action((x, c) => c.copy(model_pt = x))
      opt[String]("cp_dir")
        .text("directory of checkpoint")
        .action((x, c) => c.copy(cp_dir = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | bin/spark-submit --class bda.example.tree.RunSparkGradientBoost \
          |   out/artifacts/*/*.jar \
          |   --impurity "Variance" --loss "SquaredError" \
          |   --max_depth 10 --max_bins 32 \
          |   --min_samples 10000 --min_node_size 15 \
          |   --min_info_gain 1e-6 --num_iter 50\
          |   --learn_rate 0.02 --min_step 1e-5 \
          |   --train_pt hdfs://bda00:8020/user/houjp/data/YourTrainingData/
          |   --valid_pt hdfs://bda00:8020/user/houjp/data/YourValidationData/
          |   --model_pt hdfs://bda00:8020/user/houjp/model/YourModelName/
          |   --cp_dir hdfs://bda00:8020/user/houjp/checkpoint/
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"Spark Gradient Boost Training").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(params.cp_dir)

    // Load and parse the data file
    val (train_data, train_fs_num) = LibSVMFile.readAsReg(sc, params.train_pt)
    val (valid_data, valid_fs_num) = params.valid_pt.isEmpty match {
      case true => (None, 0)
      case false => {
        val (data, num) = LibSVMFile.readAsReg(sc, params.valid_pt)
        (Some(data), Some(num))
      }
    }

    val dt_model = GradientBoost.train(train_data,
      valid_data,
      params.impurity,
      params.loss,
      params.max_depth,
      params.max_bins,
      params.min_samples,
      params.min_node_size,
      params.min_info_gain,
      params.num_iter,
      params.learn_rate,
      params.min_step)

    if (!params.model_pt.isEmpty) {
      dt_model.save(sc, params.model_pt)
    }
  }
}