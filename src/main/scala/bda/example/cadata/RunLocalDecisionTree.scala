package bda.example.cadata

import bda.local.reader.Points
import bda.local.model.tree.{DecisionTree, DecisionTreeModel}
import bda.example.{input_dir, output_dir}
import org.apache.log4j.{Level, Logger}

/**
  * An example app for DecisionTree on cadata data set in
  * standalone(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
  * The cadata dataset can ben found at `testData/regression/cadata/`.
  */
object RunLocalDecisionTree {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val data_dir: String = input_dir + "regression/cadata/"
    val impurity: String = "Variance"
    val loss: String = "SquaredError"
    val max_depth: Int = 10
    val max_bins: Int = 32
    val bin_samples: Int = 10000
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val row_rate: Double = 1
    val col_rate: Double = 1
    val model_pt = output_dir + "dtree.model"

    val train_points = Points.readLibSVMFile(data_dir + "cadata.train", is_class = false)
    val test_points = Points.readLibSVMFile(data_dir + "cadata.test", is_class = false)

    val model: DecisionTreeModel = DecisionTree.train(train_points,
      test_points,
      impurity,
      loss,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate)

    model.save(model_pt)
  }
}