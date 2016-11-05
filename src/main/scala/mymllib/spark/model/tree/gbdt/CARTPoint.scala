package mymllib.spark.model.tree.gbdt

import org.apache.spark.rdd.RDD

/**
  * Case class which stored labels of K classes and binned features of a data point.
  *
  * @param y_K        labels of K classes of data point
  * @param binned_fs  binned features of a data point
  */
private[gbdt] case class CARTPoint(id: String,
                                   y_K: Array[Double],
                                   binned_fs: Array[Int]) extends Serializable {

  /**
    * Method to convert the point to an instance of [[String]].
    *
    * @return an instance of [[String]] which represent the point
    */
  override def toString = {
    s"id($id),y_K(${y_K.mkString(",")}),binned_fs(${binned_fs.zipWithIndex.mkString(",")})"
  }
}

/**
  * Static methods of [[CARTPoint]].
  */
private[gbdt] object CARTPoint {

  /**
    * Method to convert data set which represented as [[GBDTPoint]] to a RDD of [[CARTPoint]].
    *
    * @param gbdt_ps  a RDD of [[GBDTPoint]]
    * @return         a RDD of [[CARTPoint]]
    */
  def toCARTPoint(gbdt_ps: RDD[GBDTPoint]): RDD[CARTPoint] = {
    gbdt_ps.map {
      gbdt_p =>
        CARTPoint.toCARTPoint(gbdt_p)
    }
  }

  /**
    * Method to convert a data point of [[GBDTPoint]] to a point of [[CARTPoint]].
    *
    * @param gbdt_p a point of [[GBDTPoint]]
    * @return       an instance of [[CARTPoint]]
    */
  def toCARTPoint(gbdt_p: GBDTPoint): CARTPoint = {

    val exp_f_K = gbdt_p.f_K.map(math.exp)
    val sum_exp = exp_f_K.sum
    val p_K = exp_f_K.map(_ / sum_exp)
    val y_K = p_K.zipWithIndex.map {
      case (p: Double, id: Int) =>
        gbdt_p.label == id match {
          case true => 1.0 - p
          case false => 0.0 - p
        }
    }
    new CARTPoint(gbdt_p.id, y_K, gbdt_p.binned_fs)
  }
}