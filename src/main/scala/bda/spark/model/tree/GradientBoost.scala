package bda.spark.model.tree

import bda.common.obj.RegPoint
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{Msg, Timer}
import bda.common.Logging
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import bda.spark.model.tree.Impurity._
import bda.spark.model.tree.Loss._

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
  def train(train_data: RDD[RegPoint],
            valid_data: Option[RDD[RegPoint]] = None,
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

    new GradientBoostTrainer(Impurity.fromString(impurity),
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
private[tree] class GradientBoostTrainer(impurity: Impurity,
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
   * @param train_data Training data which represented as a RDD of [[bda.common.obj.RegPoint]].
   * @param valid_data Validation data which represented as a RDD of [[bda.common.obj.RegPoint]] and can be none.
   * @return a [[bda.spark.model.tree.GradientBoostModel]] instance.
   */
  def train(train_data: RDD[RegPoint],
            valid_data: Option[RDD[RegPoint]]): GradientBoostModel = {
    val loss_calculator = this.loss_calculator
    val wk_learners = new Array[DecisionTreeNode](num_iter)

    // persist input RDD for reusing
    train_data.persist()
    if (!valid_data.isEmpty) {
      valid_data.get.persist()
    }
    // input.count()

    var cost_time = 0.0
    var cost_count = 0

    val timer = new Timer()
    var pre_time_cost = 0L

    // get data to train DTree
    var data = train_data

    // build weak learner 0
    val wl0 = new DecisionTreeTrainer(impurity,
      loss,
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain).train(data)
    wk_learners(0) = wl0.root

    // compute prediction and RMSE for train data
    var train_pred_err = computePredictAndError(train_data, learn_rate, wl0.root, loss_calculator).persist()
    var train_err = math.sqrt(train_pred_err.values.mean())

    // compute prediction and RMSE for validation data
    var valid_pred_err = valid_data.map { data =>
      computePredictAndError(data, learn_rate, wl0.root, loss_calculator).persist()
    }
    var valid_err = valid_pred_err.map { pred_err =>
      math.sqrt(pred_err.values.mean())
    }

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
    // Log.log("INFO", s"fitting: iter = 0, error = $min_err, cost_time = $now_time_cost")

    var iter = 1
    while (iter < num_iter) {
      val begin_t = System.nanoTime()

      // get data to train DTree
      data = train_pred_err.zip(train_data).map { case ((pred, _), lp) =>
        RegPoint(-1.0 * loss_calculator.gradient(pred, lp.label), lp.fs)
      }

      // building weak leaner #iter
      val wl = new DecisionTreeTrainer(impurity,
        loss,
        max_depth,
        max_bins,
        min_samples,
        min_node_size,
        min_info_gain).train(data)
      wk_learners(iter) = wl.root

      // compute prediction and error for train data
      val train_pre_pred_err = train_pred_err
      train_pred_err = updatePredictAndError(
        train_data,
        train_pre_pred_err,
        learn_rate,
        wl.root,
        loss_calculator).persist()
      if (iter % 20 == 0) {
        train_pred_err.checkpoint()
      }
      train_pre_pred_err.unpersist()
      if (iter % 20 == 0) {
        train_pred_err.checkpoint()
      }
      train_pre_pred_err.unpersist()
      val train_err = math.sqrt(train_pred_err.values.mean())

      // compute prediction and error for validatoin data
      val valid_pre_pred_err = valid_pred_err
      valid_pred_err = valid_data.map { data =>
        updatePredictAndError(
          data,
          valid_pre_pred_err.get,
          learn_rate,
          wl.root,
          loss_calculator).persist()
      }
      if (iter % 20 == 0) {
        valid_pred_err.foreach(_.checkpoint())
      }
      valid_pre_pred_err.foreach(_.unpersist())
      val valid_err = valid_pred_err.map { pred_err =>
        math.sqrt(pred_err.values.mean())
      }

      tol_time_cost = timer.cost()
      now_time_cost = tol_time_cost - pre_time_cost
      pre_time_cost = tol_time_cost
      cost_count += 1

      // show logs
      msg = Msg("Iter" -> cost_count, "RMSE(train)" -> train_err)
      valid_err.foreach(msg.append("RMSE(valid)", _))
      msg.append("time cost", now_time_cost)
      logInfo(msg.toString)
      // Log.log("INFO", s"fitting: iter = $iter, error = $train_err, cost_time = $now_time_cost")

      if (min_err - train_err < min_step) {

        logInfo(s"Gradient Boost model training done, average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")
        return new GradientBoostModel(wk_learners.slice(0, best_iter),
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
      } else if (train_err < min_err) {
        min_err = train_err
        best_iter = iter + 1
      }

      iter += 1
    }

    logInfo(s"GBoost model training done, average cost time of each iteration: ${tol_time_cost / cost_count}(${tol_time_cost} / ${cost_count}})")

    // unpersist input RDD
    train_data.unpersist()
    valid_data.foreach(_.unpersist())

    new GradientBoostModel(wk_learners.slice(0, best_iter),
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
   * Predict values and get the mean error for the given data using the model trained and the model weight.
   *
   * @param data A RDD of [[bda.common.obj.RegPoint]] stored true label and features.
   * @param weight Model weight
   * @param root Root node in decision tree struction.
   * @param loss Loss calculator used in gradient boosting.
   * @return RDD stored prediction and the mean error.
   */
  def computePredictAndError(data: RDD[RegPoint],
                             weight: Double,
                             root: DecisionTreeNode,
                             loss: LossCalculator): RDD[(Double, Double)] = {
    data.map { lp =>
      val pred = DecisionTreeModel.predict(lp.fs, root)
    val err = loss.computeError(lp.label, pred)
      (pred, err)
    }
  }

  /**
   * Update predictions and get the mean error for the given data using the model trained and the model weight.
   *
   * @param data A RDD of [[bda.common.obj.RegPoint]] stored true label and features.
   * @param pred_err
   * @param weight
   * @param root
   * @param loss_calculator
   * @return
   */
  def updatePredictAndError(data: RDD[RegPoint],
                            pred_err: RDD[(Double, Double)],
                            weight: Double,
                            root: DecisionTreeNode,
                            loss_calculator: LossCalculator): RDD[(Double, Double)] = {
    data.zip(pred_err).mapPartitions { iter =>
      iter.map { case (lp, (pred, err)) =>
        val new_pred = pred + DecisionTreeModel.predict(lp.fs, root) * weight
        val new_err = loss_calculator.computeError(lp.label, new_pred)
        (new_pred, new_err)
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
private[tree] class GradientBoostModel(wk_learners: Array[DecisionTreeNode],
                  impurity: Impurity,
                  loss: Loss,
                  max_depth: Int,
                  max_bins: Int,
                  min_samples: Int,
                  min_node_size: Int,
                  min_info_gain: Double,
                  num_iter: Int,
                  learn_rate: Double,
                  min_step: Double,
                  impurity_calculator: ImpurityCalculator,
                  loss_calculator: LossCalculator) extends Serializable {

  /**
   * Predict values for the given data set using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input A RDD of [[bda.common.obj.RegPoint]] stored true label and features.
   * @return RDD stored prediction.
   */
  def predict(input: RDD[RegPoint]): (RDD[Double], Double) = {
    val wk_learners = this.wk_learners
    val learn_rate = this.learn_rate
    val loss = loss_calculator

    val pred_err = input.map { case lp =>
      val pred = GradientBoostModel.predict(lp.fs, wk_learners, learn_rate)
      val err = loss.computeError(pred, lp.label)
      (pred, err)
    }

    val err = pred_err.values.mean()

    //Log.log("INFO", s"predict done, with RMSE = ${math.sqrt(err)}")

    (pred_err.keys, math.sqrt(err))
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
