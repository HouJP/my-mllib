package bda.spark.model.tree.cart

private[tree] case class CARTBin(low_split: CARTSplit, high_split: CARTSplit) {

  override def toString = {
    s"low_split($low_split), hight_split($high_split)"
  }
}