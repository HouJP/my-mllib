package bda.local.ml.model

import bda.local.ml.strategy.DTreeStrategy

/**
 * Dataset metadata and strategy for decision tree
 *
 * @param numFeatures   number of features
 * @param numData       number of instances
 * @param dTreeStrategy strategy for decision tree
 */
class DTreeMetadata(
    val numFeatures: Int,
    val numData: Int,
    val dTreeStrategy: DTreeStrategy) {

}

object DTreeMetadata {

  /**
   * Construct a [[DTreeMetadata]] instance for this dataset and parameters
   *
   * @param input training dataset
   * @param dTreeStrategy strategy for decision tree
   * @return a [[DTreeMetadata]] instance
   */
  def build(input: Array[LabeledPoint], dTreeStrategy: DTreeStrategy): DTreeMetadata = {
    val numFeatures = input(0).features.size
    val numData = input.length

    new DTreeMetadata(numFeatures, numData, dTreeStrategy)
  }
}