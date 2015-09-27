package bda.local.ml

import bda.local.ml.loss.Loss
import bda.local.ml.strategy.GBoostStrategy
import bda.local.ml.model.{GBoostModel, DTreeModel, LabeledPoint}
import bda.local.ml.util.Log

class GBoostTrainer(
    val gBoostStrategy: GBoostStrategy) {

  def fit(input: Array[LabeledPoint]): GBoostModel = {
    val numIterations = gBoostStrategy.numIterations
    val learningRate = gBoostStrategy.learningRate
    val dTreeStrategy = gBoostStrategy.dTreeStrategy
    val loss = gBoostStrategy.loss
    val minStep = gBoostStrategy.minStep
    val size = input.length
    var data= input

    val weakLeaners = new Array[DTreeModel](numIterations)

    // building weak learner #0
    val wl0 = new DTreeTrainer(dTreeStrategy).fit(data)
    weakLeaners(0) = wl0

    // compute prediction and error
    val predError = GBoostModel.computeInitialPredictAndError(input, wl0, learningRate, loss)
    var pred = predError._1
    var error = predError._2

    // compute mean error
    var minError = error.sum / size
    var bestIter = 1

    Log.log("INFO", s"Iter = 0, error = $minError")

    data = pred.zip(input).map { case (predict, lp) =>
      LabeledPoint(-1.0 * loss.gradient(predict, lp.label), lp.features)
    }

    var iter = 1
    while (iter < numIterations) {
      // building weak leaner #iter
      val wl = new DTreeTrainer(dTreeStrategy).fit(data)
      weakLeaners(iter) = wl

      // compute prediction and error
      val predError = GBoostModel.updatePredictAndError(input, pred, wl, learningRate, loss)
      pred = predError._1
      error = predError._2

      // compute mean error
      val currentError = error.sum / size

      Log.log("INFO", s"fitting: Iter = $iter, error = $currentError")

      if (minError - currentError < minStep) {
        return new GBoostModel(weakLeaners.slice(0, bestIter), gBoostStrategy)
      } else if (currentError < minError) {
        minError = currentError
        bestIter = iter + 1
      }

      // update data with pseudo-residuals
      data = pred.zip(input).map { case (predict, lp) =>
        LabeledPoint(-1.0 * loss.gradient(predict, lp.label), lp.features)
      }

      iter += 1
    }
    new GBoostModel(weakLeaners.slice(0, bestIter), gBoostStrategy)
  }
}

object GBoostTrainer {

}