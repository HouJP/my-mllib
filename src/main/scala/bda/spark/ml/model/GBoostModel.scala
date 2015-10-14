package bda.spark.ml.model

import bda.common.linalg.immutable.SparseVector
import bda.local.ml.loss.LossCalculator
import bda.local.ml.model.{DTreeModel, LabeledPoint}
import bda.spark.ml.para.GBoostPara
import org.apache.spark.rdd.RDD
import bda.local.ml.util.Log

/**
 * Class of GBoost model which stored GBoost model structure and parameters.
 *
 * @param wk_learners weak learners which formed gradient boosting model.
 * @param gb_para the configuration parameters for gradient boosting algorithm
 */
class GBoostModel(
    wk_learners: Array[DTreeModel],
    gb_para: GBoostPara) extends Serializable {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point.
   * @return predicted value from the trained model.
   */
  def predict(fs: SparseVector[Double]): Double = {
    val preds = wk_learners.map(_.predict(fs))
    preds.map(_ * gb_para.learn_rate).sum
  }

  /**
   * Predict values for the given data using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input Array of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points.
   * @return RDD stored prediction
   */
  def predict(input: RDD[LabeledPoint]): RDD[Double] = {
    val loss = gb_para.loss_calculator

    val pred_err = input.map { case lp =>
        val pred = predict(lp.features)
        val err = loss.computeError(pred, lp.label)
      (pred, err)
    }

    val err = pred_err.values.mean()

    Log.log("INFO", s"predict done, with RMSE = ${math.sqrt(err)}")

    pred_err.keys
  }
}

object GBoostModel {

  /**
   * Predict values and get the mean error for the given data using the model trained and the model weight.
   *
   * @param data RDD of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points.
   * @param weight model weight
   * @param dt_model a [[bda.local.ml.model.DTreeModel]] instance.
   * @param loss loss function used in gradient boosting.
   * @return RDD stored prediction and the mean error
   */
  def computePredictAndError(
      data: RDD[LabeledPoint],
      weight: Double,
      dt_model: DTreeModel,
      loss: LossCalculator): RDD[(Double, Double)] = {
    data.map { lp =>
      val pred = dt_model.predict(lp.features) * weight
      val err = loss.computeError(lp.label, pred)
      (pred, err)
    }
  }

  /**
   * Update the pre-predictions and get the mean error for the given data using the model trained and the model weight.
   *
   * @param data RDD of [[bda.local.ml.model.LabeledPoint]] represent true label and features of data points.
   * @param pred_err pre-predictions
   * @param weight model weight
   * @param dt_model a [[bda.local.ml.model.DTreeModel]] instance
   * @param loss loss function used in grdient boosting
   * @return RDD stored prediction and the mean error
   */
  def updatePredictAndError(
      data: RDD[LabeledPoint],
      pred_err: RDD[(Double, Double)],
      weight: Double,
      dt_model: DTreeModel,
      loss: LossCalculator): RDD[(Double, Double)] = {
    data.zip(pred_err).mapPartitions { iter =>
      iter.map { case (lp, (pred, err)) =>
        val new_pred = pred + dt_model.predict(lp.features) * weight
        val new_err = loss.computeError(lp.label, new_pred)
        (new_pred, new_err)
      }
    }
  }
}