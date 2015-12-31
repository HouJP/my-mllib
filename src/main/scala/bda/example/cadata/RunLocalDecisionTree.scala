package bda.example.cadata

import bda.local.reader.Points
import bda.local.model.tree.{DecisionTree, DecisionTreeModel}
import bda.example.{input_dir, output_dir}

/**
  * An example app for DecisionTree on cadata data set in
  * standalone(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
  * The cadata dataset can ben found at `testData/regression/cadata/`.
  */
object RunLocalDecisionTree {

  def main(args: Array[String]) {

    val data_dir: String = input_dir + "regression/cadata/"
    val feature_num: Int = 8
    val impurity: String = "Variance"
    val loss: String = "SquaredError"
    val max_depth: Int = 10
    val max_bins: Int = 32
    val bin_samples: Int = 10000
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val row_rate: Double = 1
    val col_rate: Double = 0.6
    val model_pt = output_dir + "dtree.model"

    val train_points = Points.readLibSVMFile(data_dir + "cadata.train")
    val test_points = Points.readLibSVMFile(data_dir + "cadata.test")

    val model: DecisionTreeModel = DecisionTree.train(train_points,
      test_points,
      feature_num,
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