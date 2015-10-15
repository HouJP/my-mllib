package bda.local.ml.para

/**
 * Enum to select the impurity type for the algorithm
 */
object Impurity extends Enumeration {
  type Impurity = Value
  val Variance = Value

  def fromString(name: String): Impurity = name match {
    case "Variance" | "variance" => Variance
    case _ => throw new IllegalArgumentException(s"Did not recognize Impurity name: $name")
  }
}