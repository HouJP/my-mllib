package bda.spark.ml.para

/**
 * Enum to select the loss type for the algorithm
 */
object Loss extends Enumeration {
  type Loss = Value
  val SquaredError = Value

  def fromString(name: String): Loss = name match {
    case "SquaredError" | "squared_error" => SquaredError
    case _ => throw new IllegalArgumentException(s"Did not recognize Loss name: $name")
  }
}