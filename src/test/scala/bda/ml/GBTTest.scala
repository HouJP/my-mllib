package bda.ml

import bda.local.ml.strategy.{GBoostStrategy, DTreeStrategy}
import bda.local.ml.{DTreeTrainer, GBoostTrainer}
import bda.local.ml.util.MLUtils

object GBTTest {

  def main(args: Array[String]) {
    val train_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata.part1"
    val test_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata.part2"

    val train_lp = MLUtils.loadLibSVMFile(train_path)
    val test_lp = MLUtils.loadLibSVMFile(test_path)

    val gBoostStrategy = GBoostStrategy.default

    val gBoostModel = new GBoostTrainer(gBoostStrategy).fit(train_lp)

    gBoostModel.predict(test_lp)
  }
}