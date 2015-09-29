package bda.ml

import bda.local.ml.para.{GBoostPara}
import bda.local.ml.{GBoost}
import bda.local.ml.util.MLUtils

object GBTTest {

  def main(args: Array[String]) {
    val train_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata.part1"
    val test_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata.part2"

    val train_lp = MLUtils.loadLibSVMFile(train_path)
    val test_lp = MLUtils.loadLibSVMFile(test_path)

    val gb_para = GBoostPara.default

    val gb_model = new GBoost(gb_para).fit(train_lp)

    gb_model.predict(train_lp)
    gb_model.predict(test_lp)
  }
}