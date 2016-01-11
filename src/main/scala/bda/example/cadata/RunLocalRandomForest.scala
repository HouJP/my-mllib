package bda.example.cadata

import bda.local.reader.Points
import bda.local.model.tree.{RandomForestModel, RandomForest}
import bda.example.{input_dir, output_dir}
import org.apache.log4j.{Level, Logger}

/**
 * An example app for random forest on cadata data set(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
 * The cadata dataset can ben found at `testData/regression/cadata/`.
 */
object RunLocalRandomForest {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val data_dir: String = input_dir + "regression/cadata/"
    val impurity: String = "Variance"
    val loss: String = "SquaredError"
    val max_depth: Int = 10
    val max_bins: Int = 32
    val min_samples: Int = 10000
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val row_rate: Double = 0.6
    val col_rate: Double = 0.6
    val num_trees: Int = 20
    val model_pt = output_dir + "rf.model"

    val train = Points.readLibSVMFile(data_dir + "cadata.train")
    val test = Points.readLibSVMFile(data_dir + "cadata.test")

    val model: RandomForestModel = RandomForest.train(train,
      test,
      impurity,
      loss,
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees)

    model.save(model_pt)
  }
}