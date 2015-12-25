package bda.local.model.tree

import bda.common.obj.LabeledPoint
import bda.common.util.io
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{Msg, Timer}
import bda.common.Logging
import bda.local.model.tree.Impurity._
import bda.local.model.tree.Loss._
import bda.local.evaluate.Regression.RMSE

/**
 * External interface of GBDT in standalone.
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
   * @param min_node_size Minimum number of instances in the leaf, default is 15.
   * @param min_info_gain Minimum information gain while spliting, default is 1e-6.
   * @param num_iter Number of iterations.
   * @param learn_rate Value of learning rate.
   * @param min_step Minimum step of each iteration, or stop it.
   * @return a [[bda.local.model.tree.GradientBoostModel]] instance.
   */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint] = null,
            feature_num: Int = 0,
            impurity: String = "Variance",
            loss: String = "SquaredError",
            max_depth: Int = 10,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            num_iter: Int = 50,
            learn_rate: Double = 0.02,
            min_step: Double = 1e-5): GradientBoostModel = {

    new GradientBoostTrainer(feature_num,
      Impurity.fromString(impurity),
      Loss.fromString(loss),
      max_depth,
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
   * Method to train a gradient boosting model over a training data which represented as an array of [[bda.common.obj.LabeledPoint]].
   *
   * @param train_data Training data points.
   * @param valid_data Validation data points.
   * @return a [[bda.local.model.tree.GradientBoostModel]] instance which can be used to predict.
   */
  def train(train_data: Seq[LabeledPoint], valid_data: Seq[LabeledPoint]): GradientBoostModel = {
    val size = train_data.length
    var train_new_data = train_data

    val wk_learners = new Array[DecisionTreeNode](num_iter)

    var cost_count = 0
    val timer = new Timer()
    var pre_time_cost = 0L

    // building weak learner #0
    val wl0 = new DecisionTreeTrainer(feature_num,
      impurity,
      loss,
      max_depth,
      min_node_size,
      min_info_gain).train(train_new_data, null)
    wk_learners(0) = wl0.root

    // compute prediction and error of training dataset
    var train_pred = wl0.computePredict(train_data, 1.0)
    var train_rmse = evaluate(train_data.map(_.label), train_pred)

    // compute prediction and error of validation dataset
    var valid_pred = if (null != valid_data) {
      wl0.computePredict(valid_data, 1.0)
    } else {
      null
    }
    var valid_rmse = if (null != valid_data) {
      evaluate(valid_data.map(_.label), valid_pred)
    } else {
      null
    }

    var min_err = train_rmse
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

    train_new_data = train_pred.zip(train_data).map { case (predict, lp) =>
      LabeledPoint(-1.0 * loss_calculator.gradient(predict, lp.label), lp.fs)
    }

    var iter = 1
    while (iter < num_iter) {
      val begin_t = System.nanoTime()

      // building weak leaner #iter
      val wl = new DecisionTreeTrainer(feature_num,
        impurity,
        loss,
        max_depth,
        min_node_size,
        min_info_gain).train(train_new_data, null)
      wk_learners(iter) = wl.root

      // compute prediction and error for training data
      train_pred = wl.updatePredict(train_data, train_pred, learn_rate)
      train_rmse = evaluate(train_data.map(_.label), train_pred)

      // compute prediction and error for validation data
      if (null != valid_data) {
        valid_pred = wl.updatePredict(valid_data, valid_pred, learn_rate)
        valid_rmse = evaluate(valid_data.map(_.label), valid_pred)
      }

      // compute mean error
      val current_err = train_rmse

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

      if (min_err - current_err < min_step) {
        println(s"min_err = $min_err, current_err = $current_err")
        logInfo(s"Gradient Boost model training done, average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")
        return new GradientBoostModel(wk_learners.slice(0, best_iter),
          feature_num,
          impurity,
          loss,
          max_depth,
          min_node_size,
          min_info_gain,
          num_iter,
          learn_rate,
          min_step,
          impurity_calculator,
          loss_calculator)
      } else if (current_err < min_err) {
        min_err = current_err
        best_iter = iter + 1
      }

      // update data with pseudo-residuals
      train_new_data = train_pred.zip(train_data).map { case (predict, lp) =>
        LabeledPoint(-1.0 * loss_calculator.gradient(predict, lp.label), lp.fs)
      }

      iter += 1
    }

    logInfo(s"GBoost model training done, average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")
    new GradientBoostModel(wk_learners.slice(0, best_iter),
      feature_num,
      impurity,
      loss,
      max_depth,
      min_node_size,
      min_info_gain,
      num_iter,
      learn_rate,
      min_step,
      impurity_calculator,
      loss_calculator)
  }

  /**
   * Evaluate the RMSE of estimated parameters.
   *
   * @param ls labels of the data.
   * @param ps predictions of the data.
   * @return the RMSE.
   */
  def evaluate(ls: Seq[Double], ps: Seq[Double]): Double = {
    val lps = ls.zip(ps)
    RMSE(lps)
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
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param num_iter Number of iterations.
 * @param learn_rate Value of learning rate.
 * @param min_step Minimum step of each iteration, or stop it.
 * @param impurity_calculator Impurity calculator.
 * @param loss_calculator Loss calculator.
 */
@SerialVersionUID(1L)
class GradientBoostModel(val wk_learners: Array[DecisionTreeNode],
                         val feature_num: Int,
                         val impurity: Impurity,
                         val loss: Loss,
                         val max_depth: Int,
                         val min_node_size: Int,
                         val min_info_gain: Double,
                         val num_iter: Int,
                         val learn_rate: Double,
                         val min_step: Double,
                         val impurity_calculator: ImpurityCalculator,
                         val loss_calculator: LossCalculator) extends Serializable {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point
   * @return predicted value from the trained model
   */
  def predict(fs: SparseVector[Double]): Double = {
    val preds = wk_learners.map(DecisionTreeModel.predict(fs, _))
    preds.map(_ * learn_rate).sum + preds(0) * (1.0 - learn_rate)
  }

  /**
   * Predict values for the given data using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input Array of [[bda.common.obj.LabeledPoint]] represent true label and features of data points
   * @return Array stored prediction
   */
  def predict(input: Array[LabeledPoint]): (Array[Double], Double) = {
    val err_counter = loss match {
      case SquaredError => new SquaredErrorCounter()
      case _ => throw new IllegalArgumentException(s"Did not recognize loss type: ${loss}")
    }

    val output = input.map { p =>
      val pre = predict(p.fs)
      err_counter :+= (pre, p.label)
      pre
    }

    //Log.log("INFO", s"predict done, with RMSE = ${err_counter.getMean}")

    (output, err_counter.getMean)
  }

  /**
   * Store GBDT model on the disk.
   *
   * @param pt Path of the location on the disk.
   */
  def save(pt: String): Unit = {

    io.writeObject[GradientBoostModel](pt, this)
  }
}

object GradientBoostModel {

  /**
   * Load GBDT model from the disk.
   *
   * @param pt The directory of the GBDT model.
   * @return A [[bda.local.model.tree.GradientBoostModel]] instance.
   */
  def load(pt: String): GradientBoostModel = {

    io.readObject[GradientBoostModel](pt)
  }
}
