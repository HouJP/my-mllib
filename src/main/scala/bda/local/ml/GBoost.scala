package bda.local.ml

import bda.local.ml.Strategy.GBoostStrategy
import bda.local.ml.model.LabeledPoint

class GBoost(private val gBoostStrategy: GBoostStrategy) {
  def run(input: Array[LabeledPoint]): Unit = {

  }
}

object GBoost {
  def train(input: Array[LabeledPoint], gBoostStrategy: GBoostStrategy): Unit = {
    new GBoost(gBoostStrategy).run(input)
  }
}