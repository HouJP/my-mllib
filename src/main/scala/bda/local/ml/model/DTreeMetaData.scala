package bda.local.ml.model

import bda.local.ml.DTreeStrategy

class DTreeMetadata(
    val numFeatures: Int,
    val numData: Int,
    val dTreeStrategy: DTreeStrategy) {

}

object DTreeMetadata {
  def build(input: Array[LabeledPoint], dTreeStrategy: DTreeStrategy): DTreeMetadata = {
    val numFeatures = input(0).features.size
    val numData = input.length

    new DTreeMetadata(numFeatures, numData, dTreeStrategy)
  }
}