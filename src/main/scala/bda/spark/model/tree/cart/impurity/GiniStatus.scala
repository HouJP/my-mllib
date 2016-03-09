package bda.spark.model.tree.cart.impurity

private[cart] class GiniStatus(num: Int,
                               stt: Array[Double],
                               val map_label: Map[Double, Int],
                               val num_label: Int) extends ImpurityStatus(num, stt) {

}