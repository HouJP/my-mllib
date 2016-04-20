package bda.example.cadata

import bda.local.reader.Points
import bda.local.model.tree.{GradientBoostModel, GradientBoost}
import bda.example.{input_dir, output_dir}
import org.apache.log4j.{Level, Logger}

/**
  * An example app for GradientBoost on cadata data set(https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata).
  * The cadata dataset can ben found at `testData/regression/cadata/`.
  */
object RunLocalGradientBoost {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("aka").setLevel(Level.WARN)

    val data_dir: String = input_dir + "regression/"
    val impurity: String = "Variance"
    val loss: String = "SquaredError"
    val max_depth: Int = 10
    val max_bins: Int = 32
    val min_samples: Int = 10000
    val min_node_size: Int = 15
    val min_info_gain: Double = 1e-6
    val row_rate: Double = 1
    val col_rate: Double = 1
    val num_iter: Int = 300
    val learn_rate: Double = 0.02
    val min_step: Double = 1e-5
    val model_pt = output_dir + "gbdt.model"

    val train = Points.readLibSVMFile(data_dir + "cadata.train", is_class = false)
    val test = Points.readLibSVMFile(data_dir + "cadata.test", is_class = false)

    val model: GradientBoostModel = GradientBoost.train(train,
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
      num_iter,
      learn_rate,
      min_step)

    model.save(model_pt)
  }
}