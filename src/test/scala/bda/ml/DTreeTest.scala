package bda.ml

import bda.local.ml.para.DTreePara
import bda.local.ml.DTree
import bda.local.ml.util.MLUtils

object DTreeTest {
  def main(args: Array[String]) {
    val train_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata.part1"
    val test_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata.part2"

    val train_points = MLUtils.loadLibSVMFile(train_path)
    val test_points = MLUtils.loadLibSVMFile(test_path)

    val dt_para = DTreePara.default

    val model = new DTree(dt_para).fit(train_points)

    model.predict(train_points)
    model.predict(test_points)
  }
}