package bda.local.ml.util

/**
 * Class of Log used to show conditions.
 */
object Log {

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