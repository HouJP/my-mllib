package bda.spark.model.tree.gbdt.impurity

/**
  * Class used to record status of Variance impurity.
  *
  * @param num_label  number of different labels
  * @param num_data   number of total labels
  * @param stt        records of status of impurity
  */
private[gbdt] class VarianceStatus(num_label: Int,
                                   num_data: Int,
                                   stt: Array[Double]) extends ImpurityStatus(num_label, num_data, stt) {

}