package bda.local.runnable.decisionTree

import scopt.OptionParser
import bda.local.reader.LibSVMFile
import bda.local.model.tree.DecisionTreeModel

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

    val parser = new OptionParser[Params]("RunLocalDecisionTree") {
      head("RunLocalDTree: an example app for DecisionTree on your data.")
      opt[String]("test_pt")
        .required()
        .text("input paths to the dataset in LibSVM format")
        .action((x, c) => c.copy(test_pt = x))
      opt[String]("model_pt")
        .required()
        .text("directory of the decision tree model")
        .action((x, c) => c.copy(model_pt = x))
      opt[String]("predict_pt")
        .text("directory of the prediction result")
        .action((x, c) => c.copy(predict_pt = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | java -jar out/artifacts/*/*.jar \
          |   /user/houjp/data/YourTestDataName
          |   /user/houjp/model/YourModelName
          |   /user/houjp/data/YourOutDataName
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
    val (test_data, train_fs_num) = {
      val (data, num) = LibSVMFile.readAsReg(params.test_pt)
      (data.toArray, num)
    }

    val dt_model = DecisionTreeModel.load(params.model_pt)
    val (predicions, err) = dt_model.predict(test_data)

    // show RMSE
    println(s"Prediction done, RMSE(test_data)=$err")
  }
}