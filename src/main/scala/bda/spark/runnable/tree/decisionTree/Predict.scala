package bda.spark.runnable.decisionTree

import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import bda.spark.reader.LibSVMFile
import bda.spark.model.tree.DecisionTreeModel

/**
 * Decision Tree predictor.
 *
 * Input:
 * - test_pt format: label fid1:v1 fid2:v2 ...
 * Both label and v are doubles, fid are integers starting from 1.
 *
 * Output:
 * - predict_pt format:predicted_label
 */
object Predict {

  /** command line parameters */
  case class Params(test_pt: String = "",
                    model_pt: String = "",
                    predict_pt: String = "")

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkDecisionTree") {
      head("RunSparkDTree: an example app for DecisionTree on your data.")
      arg[String]("<test_pt>")
        .required()
        .text("input paths to the dataset in LibSVM format")
        .action((x, c) => c.copy(test_pt = x))
      arg[String]("<model_pt>")
        .required()
        .text("directory of the decision tree model")
        .action((x, c) => c.copy(model_pt = x))
      arg[String]("<predict_pt>")
        .required()
        .text("directory of the prediction result")
        .action((x, c) => c.copy(predict_pt = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | bin/spark-submit --class bda.example.tree.RunSparkDecisionTree \
          |   hdfs://bda00:8020/user/houjp/data/YourTestDataName
          |   hdfs://bda00:8020/user/houjp/model/YourModelName
          |   hdfs://bda00:8020/user/houjp/data/YourOutDataName
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"Spark Decision Tree Prediction").setMaster("local[2]")
    val sc = new SparkContext(conf)

    // Load and parse the data file
    val (test_data, train_fs_num) = LibSVMFile.readAsReg(sc, params.test_pt)

    val dt_model = DecisionTreeModel.load(sc, params.model_pt)
    val (predicions, err) = dt_model.predict(test_data)
  }
}