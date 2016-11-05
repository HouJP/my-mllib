package mymllib.spark.model.tree.cart.impurity

/**
  * Class used to record status of impurity.
  *
  * @param num number of labels
  * @param stt records of status of impurity
  */
private[cart] class ImpurityStatus(val num: Int,
                                   val stt: Array[Double]) extends Serializable {
}