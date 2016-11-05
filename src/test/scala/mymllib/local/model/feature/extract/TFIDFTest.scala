package mymllib.local.model.feature.extract

import org.scalatest.FunSuite
import bda.common.util.io

class TFIDFTest extends FunSuite {

  test("apply") {
    val text_path = "data/text/DocsCN.txt"

    val documents = io.readLines(text_path).toArray

    TFIDF(documents)
  }
}