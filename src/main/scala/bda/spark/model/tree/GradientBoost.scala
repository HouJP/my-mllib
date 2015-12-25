package bda.spark.model.tree

import bda.common.obj.LabeledPoint
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{Msg, Timer}
import bda.common.Logging
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import bda.spark.model.tree.Impurity._
import bda.spark.model.tree.Loss._
import bda.spark.evaluate.Regression.RMSE

/**
 * External interface of GBDT on spark.
 */
object GradientBoost {

  /**
   * An adapter of training a GBDT model.
   *
   * @param train_data Training data points.
   * @param valid_data Validation data points.
   * @param impurity Impurity type with String, default is "Variance".
   * @param loss Loss function type with String, default is "SquaredError".
   * @param max_depth Maximum depth of the decision tree, default is 6.
   * @param max_bins Maximum number of bins, default is 32.
   * @param min_samples Minimum number of samples used in finding splits and bins, default is 10000.
   * @param min_node_size Minimum number of instances in the leaf, default is 15.
   * @param min_info_gain Minimum information gain while spliting, default is 1e-6.
   * @param num_iter Number of iterations.
   * @param learn_rate Value of learning rate.
   * @param min_step Minimum step of each iteration, or stop it.
   * @return a [[bda.spark.model.tree.GradientBoostModel]] instance.
   */
  def train(train_data: RDD[LabeledPoint],
            valid_data: RDD[LabeledPoint] = null,
            feature_num: Int = 0,
            impurity: String = "Variance",
            loss: String = "SquaredError",
            max_depth: Int = 10,
            max_bins: Int = 32,
            min_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            num_iter: Int = 50,
            learn_rate: Double = 0.02,
            min_step: Double = 1e-5): GradientBoostModel = {

    new GradientBoostTrainer(feature_num,
      Impurity.fromString(impurity),
      Loss.fromString(loss),
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain,
      num_iter,
      learn_rate,
      min_step).train(train_data, valid_data)
  }
}

/**
 * A class which implement GBDT algorithm.
 *
 * @param impurity Impurity type with [[bda.spark.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.spark.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param max_bins Maximum number of bins.
 * @param min_samples Minimum number of samples used in finding splits and bins.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param num_iter Number of iterations.
 * @param learn_rate Value of learning rate.
 * @param min_step Minimum step of each iteration, or stop it.
 */
private[tree] class GradientBoostTrainer(feature_num: Int,
                                         impurity: Impurity,
                                         loss: Loss,
                                         max_depth: Int,
                                         max_bins: Int,
                                         min_samples: Int,
                                         min_node_size: Int,
                                         min_info_gain: Double,
                                         num_iter: Int,
                                         learn_rate: Double,
                                         min_step: Double) extends Logging {

  /** Impurity calculator */
  val impurity_calculator = impurity match {
    case Variance => VarianceCalculator
    case _ => throw new IllegalAccessException(s"Did not recognize impurity type: $impurity")
  }

  /** Loss calculator */
  val loss_calculator = loss match {
    case SquaredError => SquaredErrorCalculator
    case _ => throw new IllegalArgumentException(s"Did not recognize loss type: $loss")
  }

  /**
   * Method to train a GBDT model over training data.
   *
   * @param train_data Training data which represented as a RDD of [[bda.common.obj.LabeledPoint]].
   * @param valid_data Validation data which represented as a RDD of [[bda.common.obj.LabeledPoint]] and can be none.
   * @return a [[bda.spark.model.tree.GradientBoostModel]] instance.
   */
  def train(train_data: RDD[LabeledPoint],
            valid_data: RDD[LabeledPoint]): GradientBoostModel = {
    val loss_calculator = this.loss_calculator
    val wk_learners = new Array[DecisionTreeNode](num_iter)

    // persist input RDD for reusing
    train_data.persist()
    if (null != valid_data) {
      valid_data.persist()
    }
    // input.count()

    var cost_time = 0.0
    var cost_count = 0

    val timer = new Timer()
    var pre_time_cost = 0L

    // get data to train DTree
    var data = train_data

    // build weak learner 0
    val wl0 = new DecisionTreeTrainer(feature_num,
      impurity,
      loss,
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain).train(data, null)
    wk_learners(0) = wl0.root

    // compute prediction and RMSE for train data
    var train_pred = computePredict(train_data, learn_rate, wl0.root, loss_calculator).persist()
    var train_rmse = RMSE(train_data.map(_.label).zip(train_pred))

    // compute prediction and RMSE for validation data
    var valid_pred = if (null != valid_data) {
      computePredict(data, learn_rate, wl0.root, loss_calculator).persist()
    } else {
      null
    }
    var valid_rmse = if (null != valid_data) {
      RMSE(valid_data.map(_.label).zip(valid_pred))
    }

    var min_rmse = train_rmse
    var best_iter = 1

    var tol_time_cost = timer.cost()
    var now_time_cost = tol_time_cost - pre_time_cost
    pre_time_cost = tol_time_cost
    cost_count += 1

    // show logs
    var msg = Msg("Iter" -> cost_count, "RMSE(train)" -> train_rmse)
    if (null != valid_data) {
      msg.append("RMSE(valid)", valid_rmse)
    }
    msg.append("time cost", now_time_cost)
    logInfo(msg.toString)

    var iter = 1
    while (iter < num_iter) {
      val begin_t = System.nanoTime()

      // get data to train DTree
      data = train_pred.zip(train_data).map { case (pred, lp) =>
        LabeledPoint(-1.0 * loss_calculator.gradient(pred, lp.label), lp.fs)
      }

      // building weak leaner #iter
      val wl = new DecisionTreeTrainer(feature_num,
        impurity,
        loss,
        max_depth,
        max_bins,
        min_samples,
        min_node_size,
        min_info_gain).train(data, null)
      wk_learners(iter) = wl.root

      // compute prediction and error for train data
      val train_pre_pred = train_pred
      train_pred = updatePredict(
        train_data,
        train_pre_pred,
        learn_rate,
        wl.root,
        loss_calculator).persist()
      if (iter % 20 == 0) {
        train_pred.checkpoint()
      }
      train_pre_pred.unpersist()
      if (iter % 20 == 0) {
        train_pred.checkpoint()
      }
      train_pre_pred.unpersist()
      train_rmse = RMSE(train_data.map(_.label).zip(train_pred))


      // compute prediction and error for validatoin data
      val valid_pre_pred_err = valid_pred
      if (null != valid_data) {
        valid_pred = updatePredict(
          valid_data,
          valid_pre_pred_err,
          learn_rate,
          wl.root,
          loss_calculator).persist()
        if (0 == (iter % 20)) {
          valid_pred.localCheckpoint()
        }
        valid_pre_pred_err.unpersist()
        valid_rmse = RMSE(valid_data.map(_.label).zip(valid_pred))
      }

      tol_time_cost = timer.cost()
      now_time_cost = tol_time_cost - pre_time_cost
      pre_time_cost = tol_time_cost
      cost_count += 1

      // show logs
      msg = Msg("Iter" -> cost_count, "RMSE(train)" -> train_rmse)
      if (null != valid_data) {
        msg.append("RMSE(valid)", valid_rmse)
      }
      msg.append("time cost", now_time_cost)
      logInfo(msg.toString)
      // Log.log("INFO", s"fitting: iter = $iter, error = $train_err, cost_time = $now_time_cost")

      if (min_rmse - train_rmse < min_step) {

        logInfo(s"Gradient Boost model training done, " +
          s"average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")

        return new GradientBoostModel(wk_learners.slice(0, best_iter),
          feature_num,
          impurity,
          loss,
          max_depth,
          max_bins,
          min_samples,
          min_node_size,
          min_info_gain,
          num_iter,
          learn_rate,
          min_step,
          impurity_calculator,
          loss_calculator)
      } else if (train_rmse < min_rmse) {
        min_rmse = train_rmse
        best_iter = iter + 1
      }

      iter += 1
    }

    logInfo(s"GBoost model training done, " +
      s"average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")

    // unpersist input RDD
    train_data.unpersist()
    if (null != valid_data) {
      valid_data.unpersist()
    }

    new GradientBoostModel(wk_learners.slice(0, best_iter),
      feature_num,
      impurity,
      loss,
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain,
      num_iter,
      learn_rate,
      min_step,
      impurity_calculator,
      loss_calculator)
  }

  /**
   * Predict values for the given data using the model trained and the model weight.
   *
   * @param data A RDD of [[bda.common.obj.LabeledPoint]] stored true label and features.
   * @param weight Model weight
   * @param root Root node in decision tree struction.
   * @param loss Loss calculator used in gradient boosting.
   * @return RDD stored prediction.
   */
  def computePredict(data: RDD[LabeledPoint],
                     weight: Double,
                     root: DecisionTreeNode,
                     loss: LossCalculator): RDD[Double] = {
    data.map { lp =>
      DecisionTreeModel.predict(lp.fs, root)
    }
  }

  /**
   * Update predictions for the given data using the model trained and the model weight.
   *
   * @param data A RDD of [[bda.common.obj.LabeledPoint]] stored true label and features.
   * @param pred_err
   * @param weight
   * @param root
   * @param loss_calculator
   * @return predictions updated.
   */
  def updatePredict(data: RDD[LabeledPoint],
                    pred_err: RDD[Double],
                    weight: Double,
                    root: DecisionTreeNode,
                    loss_calculator: LossCalculator): RDD[Double] = {
    data.zip(pred_err).mapPartitions { iter =>
      iter.map { case (lp, pred) =>
         pred + DecisionTreeModel.predict(lp.fs, root) * weight
      }
    }
  }
}

/**
 * Class of GBDT model which stored GBDT model structure and parameters.
 *
 * @param wk_learners weak learners
 *                    which formed gradient boosting model
 *                    and represented as [[bda.spark.model.tree.DecisionTreeNode]].
 * @param impurity Impurity type with [[bda.spark.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.spark.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param max_bins Maximum number of bins.
 * @param min_samples Minimum number of samples used in finding splits and bins.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param num_iter Number of iterations.
 * @param learn_rate Value of learning rate.
 * @param min_step Minimum step of each iteration, or stop it.
 * @param impurity_calculator Impurity calculator.
 * @param loss_calculator Loss calculator.
 */
class GradientBoostModel(val wk_learners: Array[DecisionTreeNode],
                         val feature_num: Int,
                         val impurity: Impurity,
                         val loss: Loss,
                         val max_depth: Int,
                         val max_bins: Int,
                         val min_samples: Int,
                         val min_node_size: Int,
                         val min_info_gain: Double,
                         val num_iter: Int,
                         val learn_rate: Double,
                         val min_step: Double,
                         val impurity_calculator: ImpurityCalculator,
                         val loss_calculator: LossCalculator) extends Serializable {

  /**
   * Predict values for the given data set using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input A RDD of [[bda.common.obj.LabeledPoint]] stored true label and features.
   * @return RDD stored prediction.
   */
  def predict(input: RDD[LabeledPoint]): RDD[Double] = {
    val wk_learners = this.wk_learners
    val learn_rate = this.learn_rate
    val loss = loss_calculator

    val pred = input.map { case lp =>
      GradientBoostModel.predict(lp.fs, wk_learners, learn_rate)
    }

    pred
  }

  /**
   * Store GBDT model on the disk.
   *
   * @param sc Spark Context.
   * @param pt Path of the location on the disk.
   */
  def save(sc: SparkContext, pt: String): Unit = {

    val model_rdd = sc.makeRDD(Seq(this))
    model_rdd.saveAsObjectFile(pt)
  }
}

object GradientBoostModel {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point.
   * @param wk_learners weak learners formed by roots of decision trees.
   * @param learn_rate Value of learning rate.
   * @return Value of prediction
   */
  private[tree] def predict(fs: SparseVector[Double],
              wk_learners: Array[DecisionTreeNode],
              learn_rate: Double): Double = {
    val preds = wk_learners.map(DecisionTreeModel.predict(fs, _))
    preds.map(_ * learn_rate).sum + DecisionTreeModel.predict(fs, wk_learners(0)) * (1 - learn_rate)
  }

  /**
   * Load GBDT model from the disk.
   *
   * @param sc Spark context.
   * @param pt The directory of the GBDT model.
   * @return A [[bda.spark.model.tree.GradientBoostModel]] instance.
   */
  def load(sc: SparkContext, pt: String): GradientBoostModel = {

    sc.objectFile[GradientBoostModel](pt).first()
  }
}
