package mymllib.spark.model.tree

/**
  * Class of bin stored information of borders.
  *
  * @param low_split  left border of this bin
  * @param high_split right border of this bin
  */
private[tree] case class FeatureBin(low_split: FeatureSplit, high_split: FeatureSplit) {

  /**
    * Method to convert this instance into a [[String]].
    *
    * @return an instance of [[String]] represented this bin
    */
  override def toString = {
    s"low_split($low_split), high_split($high_split)"
  }
}