package bda.spark.model.tree.cart

private[cart] case class CARTBestSplit(weight_impurity: Double,
                                       l_impurity: Double,
                                       r_impurity: Double,
                                       l_predit: Double,
                                       r_predict: Double,
                                       l_count: Double,
                                       r_count: Double,
                                       split: CARTSplit)