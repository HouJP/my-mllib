package bda.spark.model.tree.cart

/**
  * Class of split which stored information of splitting.
  *
  * @param id_f ID of specified feature
  * @param threshold threshold used to split. Split left if feature-value < threshold, else right
  */
private[cart] class CARTSplit(val id_f: Int,
                              val threshold: Double) extends Serializable {

  /**
    * Method to convert the split into a [[String]].
    *
    * @return A instance of [[String]] which represents this split.
    */
  override def toString: String = {
    s"feature = $id_f, threshold = $threshold"
  }
}

/**
  * Static methods of [[CARTSplit]].
  */
private[cart] object CARTSplit {

  /**
    * Method to generate lowest [[CARTSplit]] for specified feature.
    *
    * @param id_f ID of specified feature
    * @return an instance of [[CARTSplit]] which represents lowest split
    */
  def lowest(id_f: Int): CARTSplit = {
    new CARTSplit(id_f, Double.MinValue)
  }

  /**
    * Method to generate highest [[CARTSplit]] for specified feature.
    *
    * @param id_f ID of specified feature
    * @return an instance of [[CARTSplit]] which represents highest split
    */
  def highest(id_f: Int): CARTSplit = {
    new CARTSplit(id_f, Double.MaxValue)
  }
}