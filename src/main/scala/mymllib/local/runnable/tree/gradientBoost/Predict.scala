package mymllib.local.runnable.tree.gradientBoost

import bda.common.obj.LabeledPoint
import mymllib.local.reader.Points.readLibSVMFile
import bda.common.util.io._
import scopt.OptionParser
import mymllib.local.model.tree.GradientBoostModel

/**
 * Gradient Boost predictor.
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

    val parser = new OptionParser[Params]("RunLocalGradientBoost") {
      head("RunLocalGradientBoost: an example app for Gradient Boost on your data.")
      opt[String]("test_pt")
        .required()
        .text("input paths to the dataset in LibSVM format")
        .action((x, c) => c.copy(test_pt = x))
      opt[String]("model_pt")
        .required()
        .text("directory of the Gradient Boost model")
        .action((x, c) => c.copy(model_pt = x))
      opt[String]("predict_pt")
        .required()
        .text("directory of the prediction result")
        .action((x, c) => c.copy(predict_pt = x))
      note(
        """
          |For example, the following command runs this app on your data set:
          |
          | java -jar out/artifacts/*/*.jar \
          |   --test_pt ... \
          |   --model_pt ... \
          |   --predict_pt ...
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {

    val model: GradientBoostModel = GradientBoostModel.load(params.model_pt)
    val points = readLines(params.test_pt).map(LabeledPoint.parse).toSeq

    val predictions = model.predict(points).zip(points).map {
      case (y, pn) => s"$y\t$pn"
    }

    writeLines(params.predict_pt, predictions)
  }
}