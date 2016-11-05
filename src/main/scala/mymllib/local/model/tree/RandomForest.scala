package mymllib.local.model.tree

import bda.common.obj.LabeledPoint
import bda.common.util.io
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{Msg, Timer}
import bda.common.Logging
import mymllib.local.model.tree.Impurity._
import mymllib.local.model.tree.Loss._
import mymllib.local.evaluate.Regression.RMSE

/**
 * External interface of Random Forest in standalone.
 */
object RandomForest {

  /**
   * An adapter of training a random forest model.
   *
   * @param train_data Training data points.
   * @param valid_data Validation data points.
   * @param impurity Impurity type with String, default is "Variance".
   * @param loss Loss function type with String, default is "SquaredError".
   * @param max_depth Maximum depth of the decision tree, default is 10.
   * @param max_bins Maximum number of bins, default is 32.
   * @param bin_samples Minimum number of samples used in finding splits and bins, default is 10000.
   * @param min_node_size Minimum number of instances in the leaf, default is 15.
   * @param min_info_gain Minimum information gain while splitting, default is 1e-6.
   * @param row_rate sample ratio of train data set.
   * @param col_rate sample ratio of features.
   * @param num_trees Number of decision trees.
   * @param silent whether to show logs of the algorithm.
   * @return a [[mymllib.local.model.tree.RandomForestModel]] instance.
   */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint] = null,
            impurity: String = "Variance",
            loss: String = "SquaredError",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            row_rate: Double = 0.6,
            col_rate: Double = 0.6,
            num_trees: Int = 20,
            silent: Boolean = false): RandomForestModel = {

    new RandomForestTrainer(Impurity.fromString(impurity),
      Loss.fromString(loss),
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees,
      silent).train(train_data, valid_data)
  }
}

/**
 * A class which implement random forest algorithm.
 *
 * @param impurity      Impurity type with [[mymllib.local.model.tree.Impurity]].
 * @param loss          Loss function type with [[mymllib.local.model.tree.Loss]].
 * @param max_depth     Maximum depth of the decision tree.
 * @param max_bins      Maximum number of bins.
 * @param bin_samples   Minimum number of samples used in finding splits and bins.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param row_rate      sample ratio of train data set.
 * @param col_rate      sample ratio of features.
 * @param num_trees     number of decision trees.
 * @param silent        whether to show logs of the algorithm.
 */
private[tree] class RandomForestTrainer(impurity: Impurity,
                                        loss: Loss,
                                        max_depth: Int,
                                        max_bins: Int,
                                        bin_samples: Int,
                                        min_node_size: Int,
                                        min_info_gain: Double,
                                        row_rate: Double,
                                        col_rate: Double,
                                        num_trees: Int,
                                        silent: Boolean) extends Logging {

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
   * Method to train a random forest model over a training data which
   * epresented as an array of [[mymllib.common.obj.LabeledPoint]].
   *
   * @param train_data Training data points.
   * @param valid_data Validation data points.
   * @return a [[mymllib.local.model.tree.RandomForestModel]] instance which can be used to predict.
   */
  def train(train_data: Seq[LabeledPoint],
            valid_data: Seq[LabeledPoint]): RandomForestModel = {
    val n_train = train_data.length
    val n_valid = valid_data match {
      case null => 0
      case _ => valid_data.length
    }

    // logging the input parameters
    val msg_para = Msg("n(train)" -> n_train,
      "n(valid)" -> n_valid,
      "impurity" -> impurity,
      "loss" -> loss,
      "max_depth" -> max_depth,
      "max_bins" -> max_bins,
      "bin_samples" -> bin_samples,
      "min_node_size" -> min_node_size,
      "min_info_gain" -> min_info_gain,
      "row_rate" -> row_rate,
      "col_rate" -> col_rate,
      "num_trees" -> num_trees)
    if (!silent) {
      logInfo(msg_para.toString)
    }

    val wk_learners = new Array[DecisionTreeNode](num_trees)
    var ind = 0
    while (ind < num_trees) {
      val timer = new Timer()

      val wl = new DecisionTreeTrainer(impurity,
        loss,
        max_depth,
        max_bins,
        bin_samples,
        min_node_size,
        min_info_gain,
        row_rate,
        col_rate,
        true).train(train_data, null)
      wk_learners(ind) = wl.root

      ind += 1

      // show messages
      val msg = Msg("Iter" -> ind)
      val train_rmse = RMSE(train_data.map { p =>
        (p.label, RandomForestModel.predict(p.fs, wk_learners.slice(0, ind)))
      })
      msg.append("RMSE(train)", train_rmse)
      if (null != valid_data) {
        val valid_rmse = RMSE(valid_data.map { p =>
          (p.label, RandomForestModel.predict(p.fs, wk_learners.slice(0, ind)))
        })
        msg.append("RMSE(valid)", valid_rmse)
      }
      val cost_time = timer.cost()
      msg.append("CostTime", cost_time + "ms")
      if (!silent) {
        logInfo(msg.toString)
      }
    }

    new RandomForestModel(wk_learners,
      impurity,
      loss,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees,
      impurity_calculator,
      loss_calculator)
  }
}

/**
 * Class of random forest model which stored random forest model structure and parameters.
 *
 * @param wk_learners   weak learners
 *                      which formed random forest model
 *                      and represented as [[mymllib.local.model.tree.DecisionTreeNode]].
 * @param impurity      Impurity type with [[mymllib.local.model.tree.Impurity]].
 * @param loss          Loss function type with [[mymllib.local.model.tree.Loss]].
 * @param max_depth     Maximum depth of the decision tree.
 * @param max_bins      Maximum number of bins.
 * @param bin_samples   Minimum number of samples used in finding splits and bins.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while splitting.
 * @param row_rate      sample ratio of train data set, default is 0.6.
 * @param col_rate      sample ratio of features, default is 0.6.
 * @param num_trees     number of decision trees.
 * @param impurity_calculator Impurity calculator.
 * @param loss_calculator Loss calculator.
 */
@SerialVersionUID(1L)
class RandomForestModel(val wk_learners: Array[DecisionTreeNode],
                        val impurity: Impurity,
                        val loss: Loss,
                        val max_depth: Int,
                        val max_bins: Int,
                        val bin_samples: Int,
                        val min_node_size: Int,
                        val min_info_gain: Double,
                        val row_rate: Double,
                        val col_rate: Double,
                        val num_trees: Int,
                        val impurity_calculator: ImpurityCalculator,
                        val loss_calculator: LossCalculator) extends Serializable {

  /**
   * Predict values for the given data set using the model trained.
   * Statistic RMSE while predicting.
   *
   * @param input A Sequence of [[mymllib.common.obj.LabeledPoint]] stored true label and features.
   * @return RDD stored prediction.
   */
  def predict(input: Seq[LabeledPoint]): Seq[Double] = {
    val wk_learners = this.wk_learners

    val pred = input.map { lp =>
      RandomForestModel.predict(lp.fs, wk_learners)
    }

    pred
  }

  /**
   * Store random forest model on the disk.
   *
   * @param pt Path of the location on the disk.
   */
  def save(pt: String): Unit = {

    io.writeObject[RandomForestModel](pt, this)
  }
}

object RandomForestModel {

  /**
    * Load random forest model from the disk.
   *
    * @param pt The directory of the random forest model.
    * @return A [[mymllib.local.model.tree.RandomForestModel]] instance.
   */
  def load(pt: String): RandomForestModel = {

    io.readObject[RandomForestModel](pt)
  }

  /**
    * Predict values for a single data point using the model trained.
   *
    * @param fs feature vector of a single data point.
    * @param wk_learners weak learners formed by roots of decision trees.
    * @return Value of prediction
   */
  private[tree] def predict(fs: SparseVector[Double],
                            wk_learners: Array[DecisionTreeNode]): Double = {
    wk_learners.map(DecisionTreeModel.predict(fs, _)).sum / wk_learners.length
  }
}
