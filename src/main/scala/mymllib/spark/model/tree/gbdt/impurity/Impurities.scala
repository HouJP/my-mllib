package mymllib.spark.model.tree.gbdt.impurity

/**
  * Enum to select the impurity type for the algorithm.
  */
private[gbdt] object Impurities {

  /**
    * Method to onvert [[String]] to [[Impurity]].
    *
    * @param s type of impurity
    * @return an instance of [[Impurity]]
    */
  def fromString(s: String): Impurity = {
    s match {
      case "Variance" | "variance" => Variance
      case _ => throw new IllegalArgumentException(s"The algorithm doesn't support this impurity type($s)")
    }
  }
}