package bda.example.abalone

import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser


/**
 * An example app of preprocessing on abalone data set(http://archive.ics.uci.edu/ml/datasets/Abalone).
 * The abalone dataset can ben found at `testData/regression/abalone.data/`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RunSparkPreprocessor {
  case class Params(input_pt: String = "/Users/hugh_627/ICT/bda/testData/regression/abalone.data",
                    output_train_pt: String = "",
                    output_test_pt: String = "")

  def main(args: Array[String]) {
    val default_params = Params()

    val parser = new OptionParser[Params]("RunSparkPreprocessor") {
      head("RunSparkPreprocessor: an example app of preprocessing on cadata data.")
      opt[String]("input_pt")
        .text(s"impurity of each node, default: ${default_params.input_pt}")
        .action((x, c) => c.copy(input_pt = x))
      opt[String]("output_train_pt")
        .text(s"loss function, default: ${default_params.output_train_pt}")
        .action((x, c) => c.copy(output_train_pt = x))
      opt[String]("output_test_pt")
        .text(s"maximum depth of tree, default: ${default_params.output_test_pt}")
        .action((x, c) => c.copy(output_test_pt = x))
      note(
        """
          |For example, the following command runs this app on the abalone dataset:
          |
          | bin/spark-submit --class bda.example.abalone.RunSparkPreprocessor \
          |   out/artifacts/*/*.jar \
          |   --input_pt hdfs://bda00:8020/user/bda/testData/regression/abalone.data \
          |   --output_train_pt hdfs://bda00:8020/user/out/train \
          |   --output_test_pt hdfs://bda00:8020/user/out/test
        """.stripMargin)
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"Preprocessor of abalone with $params")//.setMaster("local[2]")
    val sc = new SparkContext(conf)

    // Load and parse the data file
    val data = sc.textFile(params.input_pt).map { line =>
      val sub_lines = line.split(",")

      var str = sub_lines.last.toString

      for (i <- 1 until sub_lines.length - 1) {
        str += s" $i:${sub_lines(i)}"
      }
      sub_lines(0) match {
        case "F" => str += s" ${sub_lines.length - 1}:1" +
          s" ${sub_lines.length}:0" +
          s" ${sub_lines.length + 1}:0"
        case "M" => str += s" ${sub_lines.length - 1}:0" +
          s" ${sub_lines.length}:1" +
          s" ${sub_lines.length + 1}:0"
        case _ => str += s" ${sub_lines.length - 1}:0" +
          s" ${sub_lines.length}:0" +
          s" ${sub_lines.length + 1}:1"
      }

      str
    }

    val Array(train_data, test_data) = data.randomSplit(Array(0.7, 0.3))

    train_data.saveAsTextFile(params.output_train_pt)
    test_data.saveAsTextFile(params.output_test_pt)
  }
}