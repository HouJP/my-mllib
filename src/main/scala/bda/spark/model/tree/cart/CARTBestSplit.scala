package bda.spark.model.tree.cart

/**
  * Class stored information of best splitting.
  *
  * @param weight_impurity weighted impurity of left child and right child
  * @param l_impurity impurity of left child
  * @param r_impurity impurity of right child
  * @param l_predit prediction of left child
  * @param r_predict predicton of right child
  * @param l_count labels count of left child
  * @param r_count labels count of right child
  * @param split an instance of [[CARTSplit]] for best splitting
  */
private[cart] case class CARTBestSplit(weight_impurity: Double,
                                       l_impurity: Double,
                                       r_impurity: Double,
                                       l_predit: Double,
                                       r_predict: Double,
                                       l_count: Double,
                                       r_count: Double,
                                       split: CARTSplit)