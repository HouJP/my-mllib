package mymllib.spark.model.tree.rf.impurity

/**
  * Enum to select the impurity type for the algorithm.
  */
private[rf] object Impurities {

  /**
    * Method to onvert [[String]] to [[Impurity]].
    *
    * @param s type of impurity
    * @return an instance of [[Impurity]]
    */
  def fromString(s: String): Impurity = {
    s match {
      case "Gini" | "gini" => Gini
      case "Variance" | "variance" => Variance
      case _ => throw new IllegalArgumentException(s"The algorithm doesn't support this impurity type($s)")
    }
  }
}