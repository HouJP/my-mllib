package bda.ml.util

import bda.local.ml.util.MLUtils

object MLUtilsTest {

  def main (args: Array[String]) {
    val path = "/Users/hugh_627/ICT/GBoost/project/GBoost/data/train"

    val labeledPoint = MLUtils.loadLibSVMFile(path)
  }
}