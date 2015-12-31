package bda.spark.model.tree

import bda.common.obj.LabeledPoint
import bda.common.linalg.immutable.SparseVector
import bda.common.util.{Msg, Timer}
import bda.common.Logging
import bda.spark.model.tree.Impurity._
import bda.spark.model.tree.Loss._
import bda.spark.evaluate.Regression.RMSE
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * External interface of Random Forest on spark.
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
   * @param min_samples Minimum number of samples used in finding splits and bins, default is 10000.
   * @param min_node_size Minimum number of instances in the leaf, default is 15.
   * @param min_info_gain Minimum information gain while splitting, default is 1e-6.
   * @param row_rate sample ratio of train data set, default is 0.6.
   * @param col_rate sample ratio of features, default is 0.6.
   * @param num_trees Number of decision trees, default is 20.
   * @return a [[bda.spark.model.tree.RandomForestModel]] instance.
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
            row_rate: Double = 0.6,
            col_rate: Double = 0.6,
            num_trees: Int = 20): RandomForestModel = {

    new RandomForestTrainer(feature_num,
      Impurity.fromString(impurity),
      Loss.fromString(loss),
      max_depth,
      max_bins,
      min_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees).train(train_data, valid_data)
  }
}

/**
 * A class which implement random forest algorithm.
 *
 * @param impurity Impurity type with [[bda.spark.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.spark.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param max_bins Maximum number of bins.
 * @param min_samples Minimum number of samples used in finding splits and bins.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while spliting.
 * @param row_rate sample ratio of train data set.
 * @param col_rate sample ratio of features.
 * @param num_trees number of decision trees.
 */
private[tree] class RandomForestTrainer(feature_num: Int,
                                        impurity: Impurity,
                                        loss: Loss,
                                        max_depth: Int,
                                        max_bins: Int,
                                        min_samples: Int,
                                        min_node_size: Int,
                                        min_info_gain: Double,
                                        row_rate: Double,
                                        col_rate: Double,
                                        num_trees: Int) extends Logging {

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
   * epresented as an RDD of [[bda.common.obj.LabeledPoint]].
   *
   * @param train_data Training data points.
   * @param valid_data Validation data points.
   * @return a [[bda.spark.model.tree.RandomForestModel]] instance which can be used to predict.
   */
  def train(train_data: RDD[LabeledPoint], valid_data: RDD[LabeledPoint]): RandomForestModel = {
    val timer = new Timer()
    var pre_time = 0L
    var now_time = 0L
    var cost_time = 0L
    var msg: Msg = null

    val wk_learners = new Array[DecisionTreeNode](num_trees)
    var ind = 0
    while (ind < num_trees) {
      val wl = new DecisionTreeTrainer(feature_num,
        impurity,
        loss,
        max_depth,
        max_bins,
        min_samples,
        min_node_size,
        min_info_gain,
        row_rate,
        col_rate).train(train_data, null)
      wk_learners(ind) = wl.root

      ind += 1

      msg = Msg("NumOfTrees" -> ind)
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
      now_time = timer.cost()
      cost_time = now_time - pre_time
      pre_time = now_time
      msg.append("AddTime", cost_time + "ms")
      msg.append("TotalTime", now_time + "ms")
      logInfo(msg.toString)
    }

    new RandomForestModel(wk_learners,
      feature_num,
      impurity,
      loss,
      max_depth,
      max_bins,
      min_samples,
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
 * @param wk_learners weak learners
 *                    which formed random forest model
 *                    and represented as [[bda.spark.model.tree.DecisionTreeNode]].
 * @param impurity Impurity type with [[bda.spark.model.tree.Impurity]].
 * @param loss Loss function type with [[bda.spark.model.tree.Loss]].
 * @param max_depth Maximum depth of the decision tree.
 * @param min_node_size Minimum number of instances in the leaf.
 * @param min_info_gain Minimum information gain while splitting.
 * @param row_rate sample ratio of train data set, default is 0.6.
 * @param col_rate sample ratio of features, default is 0.6.
 * @param num_trees number of decision trees.
 * @param impurity_calculator Impurity calculator.
 * @param loss_calculator Loss calculator.
 */
class RandomForestModel(val wk_learners: Array[DecisionTreeNode],
                        val feature_num: Int,
                        val impurity: Impurity,
                        val loss: Loss,
                        val max_depth: Int,
                        val max_bins: Int,
                        val min_samples: Int,
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
   * @param input A RDD of [[bda.common.obj.LabeledPoint]] stored true label and features.
   * @return RDD stored prediction.
   */
  def predict(input: RDD[LabeledPoint]): RDD[Double] = {
    val wk_learners = this.wk_learners

    val pred = input.map { case lp =>
      RandomForestModel.predict(lp.fs, wk_learners)
    }

    pred
  }

  /**
   * Store random forest model on the disk.
   *
   * @param pt Path of the location on the disk.
   */
  def save(sc: SparkContext, pt: String): Unit = {

    val model_rdd = sc.makeRDD(Seq(this))
    model_rdd.saveAsObjectFile(pt)
  }
}

object RandomForestModel {

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

  /**
   * Load random forest model from the disk.
   *
   * @param pt The directory of the random forest model.
   * @return A [[bda.spark.model.tree.RandomForestModel]] instance.
   */
  def load(sc: SparkContext, pt: String): RandomForestModel = {

    sc.objectFile[RandomForestModel](pt).first()
  }
}
