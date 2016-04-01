package bda.spark.model.tree.cart.impurity

/**
  * Class used to record status of Gini impurity.
  *
  * @param num number of labels
  * @param stt records of status of impurity
  * @param map_label Map(label, index), indexed from 0
  * @param num_label number of different labels
  */
private[cart] class GiniStatus(num: Int,
                               stt: Array[Double],
                               val map_label: Map[Double, Int],
                               val num_label: Int) extends ImpurityStatus(num, stt) {

}