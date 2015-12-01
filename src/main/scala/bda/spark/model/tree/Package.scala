package bda.spark.model.tree

/**
 * Enum to select the impurity type for the algorithm.
 */
private[tree] object Impurity extends Enumeration {
  type Impurity = Value
  val Variance = Value

  def fromString(name: String): Impurity = name match {
    case "Variance" | "variance" => Variance
    case _ => throw new IllegalArgumentException(s"Did not recognize Impurity name: $name")
  }
}

/**
 * Enum to select the loss type for the algorithm.
 */
private[tree] object Loss extends Enumeration {
  type Loss = Value
  val SquaredError = Value

  def fromString(name: String): Loss = name match {
    case "SquaredError" | "squared_error" => SquaredError
    case _ => throw new IllegalArgumentException(s"Did not recognize Loss name: $name")
  }
}

/**
 * Class of Log used to show conditions.
 */
private[tree] object Log {

  /**
   * Method to show conditions.
   *
   * @param typ the type of the message
   * @param msg the message to show
   */
  def log(typ: String, msg: String): Unit = {
    println("[" + typ + "] " + msg)
  }
}