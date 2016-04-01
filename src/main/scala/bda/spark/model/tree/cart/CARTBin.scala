package bda.spark.model.tree.cart

/**
  * Class of bin stored information of borders.
  *
  * @param low_split left border of this bin
  * @param high_split right border of this bin
  */
private[tree] case class CARTBin(low_split: CARTSplit, high_split: CARTSplit) {

  /**
    * Method to convert this instance into a [[String]].
    *
    * @return an instance of [[String]] represented this bin
    */
  override def toString = {
    s"low_split($low_split), hight_split($high_split)"
  }
}