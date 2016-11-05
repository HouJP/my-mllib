package mymllib.local.util

import org.scalatest.FunSuite

class WordSegmentTest extends FunSuite {

  test("apply") {
    val s = "你好,世界"
    val ws = WordSegment(s)

    ws.foreach(println)
  }
}