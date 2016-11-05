package mymllib.spark.evaluate

import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/** Evaluation metrics for classification on Spark */
object Classification {

  private def equal(t: Double, y: Double) = math.abs(t - y) < 1e-7

  /**
    * Return the proportion of correct predictions
    * @param tys   RDD[(t: Int, y: Int)], t is label, y is prediction
    */
  def accuracy(tys: RDD[(Double, Double)]): Double = {
    val n_all = tys.count()
    require(n_all > 0, "Accuracy evaluated on Empty dataset!")
    val n_right = tys.filter {
      case (t, y) => equal(t, y)
    }.count()

    n_right.toDouble / n_all
  }

  /**
    * Compute precision rate.
    * @note When use this method, the positive or true label must be 1.
    * @param tys RDD[(trueValue, prediction)]
    * @return Precision rate
    */
  def precision(tys: RDD[(Double, Double)]): Double = {
    val positive = tys.filter(a => equal(a._2, 1)).cache()
    val n_pos = positive.count()
    val true_pos = positive.filter(a => equal(a._1, 1)).count()
    positive.unpersist(blocking = false)
    if (n_pos == 0) 1.0 else true_pos * 1.0 / n_pos
  }

  /**
    * Compute recall rate.
    * @note When use this method, the positive or true label must be 1.
    * @param tys RDD[(trueValue, prediction)]
    * @return Recall Rate.
    */
  def recall(tys: RDD[(Double, Double)]): Double = {
    val truth = tys.filter(a => equal(a._1, 1)).cache()
    val t_c = truth.count()
    val true_pos = truth.filter(a => equal(a._2, 1)).count()
    truth.unpersist(blocking = false)
    if (t_c == 0) 1.0 else true_pos * 1.0 / t_c
  }


  /**
    * Compute auc of a model.
    * Require prediction in [0,1] and trueValue be 0 or 1.
    * This is the confusion matrix.
    * |          |       |                  Truth                  |
    * |          |       | actual positive(1) | actual negative(0) |
    * |prediction|1      |   TP               |         FP         |
    * |          |0      |   FN               |         TN         |
    *
    * Roc pairs is (TPR, FPR) pairs in different thresholds.
    * TPR = TP / (TP  + FN ); FPR = Fp / (FP + TN).
    * @param tys  RDD[(trueValue, prediction)]
    * @return AUC
    */
  def auc(tys: RDD[(Double, Double)]): Double = {
    val tru = tys.filter(a => equal(a._1, 1)).cache()
    val t_c = tru.count
    val fal = tys.filter(a => equal(a._1, 0)).cache()
    val f_c = fal.count()
    //Must have both true label samples and false label samples.
    require(t_c > 0 && f_c > 0)
    val roc_pair = new ArrayBuffer[(Double, Double)]()
    for (i <- 1 until 101) {
      val tp = tru.filter(a => a._2 > i / 100.0).count()
      val fp = fal.filter(a => a._2 > i / 100.0).count()
      roc_pair.append((fp * 1.0 / f_c, tp * 1.0 / t_c))
    }
    tru.unpersist(blocking = false)
    fal.unpersist(blocking = false)
    val roc: Array[(Double, Double)] = roc_pair.reverse.toArray
    var auc = 0.0
    //calculate an area.
    for (i <- 0 until roc.length - 1) {
      auc += (roc(i + 1)._1 - roc(i)._1) * (roc(i + 1)._2 + roc(i)._2) / 2
    }
    auc += (1 - roc(roc.length - 1)._1) * (1 + roc(roc.length - 1)._2) / 2
    auc
  }
}
