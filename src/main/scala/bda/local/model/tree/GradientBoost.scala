package bda.local.model.tree

import bda.common.obj.RegPoint
import bda.common.util.io
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{Msg, Timer}
import bda.common.Logging
import bda.local.model.tree.Impurity._
import bda.local.model.tree.Loss._

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
  def train(train_data: Array[RegPoint],
            valid_data: Option[Array[RegPoint]] = None,
            impurity: String = "Variance",
            loss: String = "SquaredError",
            max_depth: Int = 10,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            num_iter: Int = 50,
            learn_rate: Double = 0.02,
            min_step: Double = 1e-5): GradientBoostModel = {

    new GradientBoostTrainer(Impurity.fromString(impurity),
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
private[tree] class GradientBoostTrainer(impurity: Impurity,
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
   * Method to train a gradient boosting model over a training data which represented as an array of [[bda.common.obj.RegPoint]].
   *
   * @param train_data Training data points.
   * @param valid_data Validation data points.
   * @return a [[bda.local.model.tree.GradientBoostModel]] instance which can be used to predict.
   */
  def train(train_data: Array[RegPoint], valid_data: Option[Array[RegPoint]]): GradientBoostModel = {
    val size = train_data.length
    var train_new_data = train_data
    var valid_new_data = valid_data

    val wk_learners = new Array[DecisionTreeNode](num_iter)

    var cost_count = 0
    val timer = new Timer()
    var pre_time_cost = 0L

    // building weak learner #0
    val wl0 = new DecisionTreeTrainer(impurity,
      loss,
      max_depth,
      min_node_size,
      min_info_gain).train(train_new_data, None)
    wk_learners(0) = wl0.root

    // compute prediction and error of training dataset
    var train_pred_err = wl0.computePredictAndError(train_data, learn_rate)
    var train_pred = train_pred_err._1
    var train_err = train_pred_err._2

    // compute prediction and error of validation dataset
    var valid_pred_err = valid_data.map(wl0.computePredictAndError(_, learn_rate))
    var valid_pred = valid_pred_err.map(_._1)
    var valid_err = valid_pred_err.map(_._2)

    var min_err = train_err
    var best_iter = 1

    var tol_time_cost = timer.cost()
    var now_time_cost = tol_time_cost - pre_time_cost
    pre_time_cost = tol_time_cost
    cost_count += 1

    // show logs
    var msg = Msg("Iter" -> cost_count, "RMSE(train)" -> train_err)
    valid_err.foreach(msg.append("RMSE(valid)", _))
    msg.append("time cost", now_time_cost)
    logInfo(msg.toString)

    train_new_data = train_pred.zip(train_data).map { case (predict, lp) =>
      RegPoint(-1.0 * loss_calculator.gradient(predict, lp.label), lp.fs)
    }

    var iter = 1
    while (iter < num_iter) {
      val begin_t = System.nanoTime()

      // building weak leaner #iter
      val wl = new DecisionTreeTrainer(impurity,
        loss,
        max_depth,
        min_node_size,
        min_info_gain).train(train_new_data, None)
      wk_learners(iter) = wl.root

      // compute prediction and error for training data
      train_pred_err = wl.updatePredictAndError(train_data, train_pred, learn_rate)
      train_pred = train_pred_err._1
      train_err = train_pred_err._2

      // compute prediction and error for validation data
      valid_pred_err = valid_data.map(wl.updatePredictAndError(_, valid_pred.get, learn_rate))
      valid_pred = valid_pred_err.map(_._1)
      valid_err = valid_pred_err.map(_._2)

      // compute mean error
      val current_err = train_err

      tol_time_cost = timer.cost()
      now_time_cost = tol_time_cost - pre_time_cost
      pre_time_cost = tol_time_cost
      cost_count += 1

      // show logs
      msg = Msg("Iter" -> cost_count, "RMSE(train)" -> train_err)
      valid_err.foreach(msg.append("RMSE(valid)", _))
      msg.append("time cost", now_time_cost)
      logInfo(msg.toString)

      if (min_err - current_err < min_step) {
        logInfo(s"Gradient Boost model training done, average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")
        return new GradientBoostModel(wk_learners.slice(0, best_iter),
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
        RegPoint(-1.0 * loss_calculator.gradient(predict, lp.label), lp.fs)
      }

      iter += 1
    }

    logInfo(s"GBoost model training done, average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")
    new GradientBoostModel(wk_learners.slice(0, best_iter),
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
private[tree] class GradientBoostModel(wk_learners: Array[DecisionTreeNode],
                                       impurity: Impurity,
                                       loss: Loss,
                                       max_depth: Int,
                                       min_node_size: Int,
                                       min_info_gain: Double,
                                       num_iter: Int,
                                       learn_rate: Double,
                                       min_step: Double,
                                       impurity_calculator: ImpurityCalculator,
                                       loss_calculator: LossCalculator) extends Serializable {

  private val serialVersionUID = 6529685098267757690L

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param fs feature vector of a single data point
   * @return predicted value from the trained model
   */
  def predict(fs: SparseVector[Double]): Double = {
    val preds = wk_learners.map(DecisionTreeModel.predict(fs, _))
    preds.map(_ * learn_rate).sum
  }

  /**
   * Predict values for the given data using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input Array of [[bda.common.obj.RegPoint]] represent true label and features of data points
   * @return Array stored prediction
   */
  def predict(input: Array[RegPoint]): (Array[Double], Double) = {
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
