package bda.example.cadata

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser

/**
 * An example app for GradientBoost on cadata data set(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
 * The cadata dataset can ben found at `testData/regression/cadata/`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RunMLLibRandomForest {
  case class Params(train_pt: String = "/Users/hugh_627/ICT/bda/testData/regression/cadata/cadata.train",
                    feature_num: Int = 8,
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
      head("RunSparkGradientBoost: an example app for Gradient Boost on cadata data.")
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
      opt[Int]("feature_num")
        .text(s"number of features, default: ${default_params.feature_num}")
        .action((x, c) => c.copy(feature_num = x))
      opt[String]("train_pt")
        .text("path to the cadata training dataset in LibSVM format")
        .action((x, c) => c.copy(train_pt = x))
      note(
        """
          |For example, the following command runs this app on the cadata dataset:
          |
          | bin/spark-submit --class bda.example.tree.RunSparkGradientBoost \
          |   out/artifacts/*/*.jar \
          |   --impurity "Variance" --loss "SquaredError" \
          |   --max_depth 10 --max_bins 32 \
          |   --min_samples 10000 --min_node_size 15 \
          |   --min_info_gain 1e-6 --num_iter 50\
          |   --learn_rate 0.02 --min_step 1e-5 \
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
    val conf = new SparkConf().setAppName(s"Spark Random Forest Training of cadata dataset").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, params.train_pt)

    // 分类数
    val numClasses = 0
    // categricalFeaturesInfo
    val categoricalFeaturesInfo = Map[Int, Int]()
    // 树的个数
    val numTrees: Int = 3
    val featureSubsetStrategy = "auto"

    //纯度计算
    val impurity = "gini"
    //树的最大层次
    val maxDepth = 4
    //特征最大装箱数
    val maxBins = 32
    //训练随机森林分类器，trainClassifier 返回的是 RandomForestModel 对象
    val model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    //    val train = Points.fromLibSVMFile(sc, params.data_dir + "/cadata.train", params.feature_num)
    //    val test = Points.fromLibSVMFile(sc, params.data_dir + "/cadata.test", params.feature_num)
    val fs = data.map(_.features)
    model.predict(fs)
  }

}