package bda.examples.ml

import scopt.OptionParser
import bda.local.ml.util.Log
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

/**
 * An example app for [[org.apache.spark.mllib.tree.GradientBoostedTrees]] on YearPredictionMSD data (http://archive.ics.uci.edu/ml/YearPredictionMSD).
 * The YearPredictionMSD dataset can ben found at `data/YearPredictionMSD/`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RunSparkGBTs {

  case class Params (
                      input: String = null,
                      cp_dir: String = null,
                      num_iter: Int = 200,
                      learn_rate: Double = 0.2,
                      loss: String = "SquaredError",
                      min_step: Double = 1e-5,
                      impurity: String = "Variance",
                      min_node_size: Int = 15,
                      max_depth: Int = 10,
                      min_info_gain: Double = 1e-5,
                      data_sample_rate: Double = 0.2,
                      max_data_sample: Int = 10000)

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkGBTs") {
      head("RunSparkGBoost: an example app for GBTs on YearPredictionMSD data.")
      opt[Int]("num_iter")
        .text(s"number of iterations, default: ${default_params.num_iter}")
        .action((x, c) => c.copy(num_iter = x))
      opt[Double]("learn_rate")
        .text(s"learning rate, default: ${default_params.learn_rate}")
        .action((x, c) => c.copy(learn_rate = x))
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
      opt[Double]("min_info_gain")
        .text(s"minimum info gain, default: ${default_params.min_info_gain}")
        .action((x, c) => c.copy(min_info_gain = x))
      opt[Double]("data_sample_rate")
        .text(s"rate to sample data, default: ${default_params.data_sample_rate}")
        .action((x, c) => c.copy(data_sample_rate = x))
      opt[Int]("max_data_sample")
        .text(s"maximum number of sampled data, default: ${default_params.max_data_sample}")
        .action((x, c) => c.copy(max_data_sample = x))
      arg[String]("<input>")
        .required()
        .text("input paths to the cadata dataset in LibSVM format")
        .action((x, c) => c.copy(input = x))
      arg[String]("<cp_dir>")
        .required()
        .text("checkpoint directory")
        .action((x, c) => c.copy(cp_dir = x))
      note(
        """
          |For example, the following command runs this app on the YearPredictionMSD dataset:
          |
          | bin/spark-submit --class bda.examples.ml.RunSparkGBTs \
          |   out/artifacts/*/*.jar \
          |   --num_iter 100 --learn_rate 0.01 \
          |   --min_node_size 15 --max_depth 10 \
          |   hdfs://bda00:8020/user/houjp/gboost/data/YearPredictionMSD/
          |   hdfs://bda00:8020/user/houjp/gboost/data/checkpoint/
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"GBTs Example with $params")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(params.cp_dir)

    // Load and parse the data file
    val train_data = MLUtils.loadLibSVMFile(sc, params.input + ".train")
    val test_data = MLUtils.loadLibSVMFile(sc, params.input + ".test")

    // Train a GradientBoostedTrees model.
    val boosting_strategy = BoostingStrategy.defaultParams("Regression")

    boosting_strategy.numIterations = params.num_iter
    boosting_strategy.learningRate = params.learn_rate
    boosting_strategy.validationTol = params.min_step

    boosting_strategy.treeStrategy.maxDepth = params.max_depth
    boosting_strategy.treeStrategy.minInfoGain = params.min_info_gain

    boosting_strategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val begin_t = System.nanoTime()
    val model = GradientBoostedTrees.train(train_data, boosting_strategy)
    val end_t = System.nanoTime()
    Log.log("INFO", s"trainning cost time: ${(end_t - begin_t) / 1e6}ms")

    // Evaluate model on train instances and compute test error
    val train_labels_pred = train_data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val train_rmse = math.sqrt(train_labels_pred.map { case(v, p) => math.pow((v - p), 2)}.mean())
    Log.log("INFO", "Train RMSE = " + train_rmse)

    // Evaluate model on test instances and compute test error
    val test_labels_pred = test_data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val test_rmse = math.sqrt(test_labels_pred.map { case(v, p) => math.pow((v - p), 2)}.mean())
    Log.log("INFO", "Test RMSE = " + test_rmse)
  }
}