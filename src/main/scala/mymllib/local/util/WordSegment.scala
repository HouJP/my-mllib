package mymllib.local.util

import org.ansj.domain.Term
import org.ansj.splitWord.analysis.ToAnalysis

/**
  * Chinese word segment using Ansj(https://github.com/NLPchina/ansj_seg)
  */
object WordSegment {

  /**
    * Segment a string into word sequence.
    *
    * @param s  chinese sentence
    * @return   word sequence
    */
  def apply(s: String): Array[String] = {
    ToAnalysis.parse(s).toArray().map {
      case t: Term => t.getName
    }
  }
}