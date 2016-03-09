package bda.spark.model.tree.cart.impurity

object Impurities {

  def fromString(s: String): Impurity = {
    s match {
      case "Gini" | "gini" => Gini
      case "Variance" | "variance" => Variance
      case _ => throw new IllegalArgumentException(s"The algorithm doesn't support this impurity type($s)")
    }
  }
}