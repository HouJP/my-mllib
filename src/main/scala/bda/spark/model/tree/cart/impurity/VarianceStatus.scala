package bda.spark.model.tree.cart.impurity

/**
  * Class used to record status of Variance impurity.
  *
  * @param num number of labels
  * @param stt records of status of impurity
  */
private[cart] class VarianceStatus(num: Int,
                                   stt: Array[Double]) extends ImpurityStatus(num, stt) {

}