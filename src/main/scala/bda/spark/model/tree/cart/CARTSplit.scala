package bda.spark.model.tree.cart

private[cart] class CARTSplit(val id_f: Int,
                              val threshold: Double) extends Serializable {

  /**
    * Method to convert this into a string.
    *
    * @return A string which represented this split.
    */
  override def toString: String = {
    s"feature = $id_f, threshold = $threshold"
  }
}

private[cart] object CARTSplit {

  def lowest(id_f: Int): CARTSplit = {
    new CARTSplit(id_f, Double.MinValue)
  }

  def highest(id_f: Int): CARTSplit = {
    new CARTSplit(id_f, Double.MaxValue)
  }
}