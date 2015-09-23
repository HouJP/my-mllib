package bda.ml

import bda.local.ml.{DTree, DTreeStrategy}
import bda.local.ml.util.MLUtils

object DTreeTest {
  def main(args: Array[String]) {
    val path = "/Users/hugh_627/ICT/GBoost/project/GBoost/data/train"

    val labeledPoint = MLUtils.loadLibSVMFile(path)

    val dTreeStrategy = DTreeStrategy.default

    val model = DTree.train(labeledPoint, dTreeStrategy)

    var rmse = 0.0
    labeledPoint.foreach { point =>
      rmse += math.pow(model.predict(point.features) - point.label, 2)
    }
    rmse /= labeledPoint.length
    rmse = math.sqrt(rmse)
    println(s"rmse = $rmse")

    model.predict(labeledPoint)
  }
}