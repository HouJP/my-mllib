package bda.local.ml.util

object Log {

  def log(typ: String, msg: String): Unit = {
    println("[" + typ + "] " + msg)
  }
}