package bda.spark.model.tree.gbdt.impurity

/**
  * Class used to record status of impurity.
  *
  * @param num_label  number of different labels
  * @param num_data   number of total labels
  * @param stt        records of status of impurity
  */
private[gbdt] class ImpurityStatus(val num_label: Int,
                                   val num_data: Int,
                                   val stt: Array[Double]) extends Serializable {

  /**
    * Convert this class to a instance of [[String]].
    *
    * @return an instance of [[String]]
    */
  override def toString: String = {
    s"num_label=$num_label,num_data=$num_data,stt=${stt.mkString(",")}"
  }
}