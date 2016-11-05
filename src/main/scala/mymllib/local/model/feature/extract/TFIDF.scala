package mymllib.local.model.feature.extract

import bda.common.util.WordSeg

object TFIDF {

  def apply(documents: Array[String]) {
    // Segment documents
    val documents_seg = documents.map(WordSeg(_))

    // Get dictionary
    val dict = documents_seg.flatten.distinct.zipWithIndex.toMap

    // Count words in documents
    val document_count = documents_seg.map {
      ws =>
        val ws_count = collection.mutable.Map[Int, Double]()
        ws.foreach {
          w: String =>
            val id = dict.getOrElse(w, -1)
            if (-1 != id) {
              ws_count(id) = ws_count.getOrElseUpdate(id, 0.0) + 1.0
            }
        }
        ws_count
    }

    // Calculate the number of documents
    val ds_n = documents.length

    // Calculate document frequency
    val ws_df = collection.mutable.Map[Int, Double]()
    document_count.foreach {
      ws =>
        ws.keys.foreach(id => ws_df(id) = ws_df.getOrElseUpdate(id, 0.0) + 1.0)
    }

    // Calculate inverse document frequency
    val ws_idf = ws_df.map {
      ks =>
        (ks._1, math.log(ds_n / (ks._2 + 1.0)))
    }

    // Calculate the tf-idf vector
    val tfidf = document_count.map {
      ws =>
        val d_len = ws.values.sum
        ws.map {
          kv =>
            (kv._1, kv._2 / d_len * ws_idf(kv._1))
        }
    }

    // Print
    tfidf.map(_.map(kv => s"${kv._1}:${kv._2}").mkString(" ")).zipWithIndex.foreach(println)
  }
}