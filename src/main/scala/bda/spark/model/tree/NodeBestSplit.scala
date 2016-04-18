package bda.spark.model.tree

/**
  * Class stored information of best splitting.
  *
  * @param weight_impurity  weighted impurity of left child and right child
  * @param l_impurity       impurity of left child
  * @param r_impurity       impurity of right child
  * @param l_predict         prediction of left child
  * @param r_predict        predicton of right child
  * @param l_count          labels count of left child
  * @param r_count          labels count of right child
  * @param split            an instance of [[FeatureSplit]] for best splitting
  */
private[tree] class NodeBestSplit(val weight_impurity: Double,
                                  val l_impurity: Double,
                                  val r_impurity: Double,
                                  val l_predict: Double,
                                  val r_predict: Double,
                                  val l_count: Double,
                                  val r_count: Double,
                                  val split: FeatureSplit) extends Serializable {
  override def toString: String = {
    s"weight_impurity($weight_impurity),l_impurity($l_impurity),r_impurity($r_impurity)," +
      s"l_predict($l_predict),r_predict($r_predict),l_count($l_count),r_count($r_count),split($split)"
  }
}