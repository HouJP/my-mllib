package bda.spark.model.tree.rf

import bda.common.obj.LabeledPoint
import bda.common.Logging
import bda.common.util.{Timer, Msg}
import bda.spark.model.tree.TreeNode
import bda.spark.model.tree.cart.CART
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * External interface of Random Forest on spark.
  */
object RandomForest extends Logging {

  /**
    * An adapter of training a random forest model.
    *
    * @param train_data Training data points
    * @param impurity Impurity type with String, default is "Variance"
    * @param max_depth Maximum depth of the decision tree, default is 10
    * @param max_bins Maximum number of bins, default is 32
    * @param bin_samples Minimum number of samples used in finding splits and bins, default is 10000
    * @param min_node_size Minimum number of instances in the leaf, default is 15
    * @param min_info_gain Minimum information gain while splitting, default is 1e-6
    * @param row_rate sample ratio of train data set, default is 0.6
    * @param col_rate sample ratio of features, default is 0.6
    * @param num_trees Number of decision trees, default is 20
    * @return a [[bda.spark.model.tree.rf.RandomForestModel]] instance
    */
  def train(train_data: RDD[LabeledPoint],
            impurity: String = "Variance",
            max_depth: Int = 10,
            max_bins: Int = 32,
            bin_samples: Int = 10000,
            min_node_size: Int = 15,
            min_info_gain: Double = 1e-6,
            row_rate: Double = 0.6,
            col_rate: Double = 0.6,
            num_trees: Int = 20): RandomForestModel = {

    val msg = Msg("n(train_data)" -> train_data.count(),
      "impurity" -> impurity,
      "max_depth" -> max_depth,
      "max_bins" -> max_bins,
      "bin_samples" -> bin_samples,
      "min_node_size" -> min_node_size,
      "min_info_gain" -> min_info_gain,
      "row_rate" -> row_rate,
      "col_rate" -> col_rate,
      "num_trees" -> num_trees
    )
    logInfo(msg.toString)

    new RandomForest(impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees).train(train_data)
  }
}

/**
  * A class which implement random forest algorithm.
  *
  * @param impurity impurity used to split node
  * @param max_depth Maximum depth of the decision tree
  * @param max_bins Maximum number of bins
  * @param bin_samples Minimum number of samples used in finding splits and bins
  * @param min_node_size Minimum number of instances in the leaf
  * @param min_info_gain Minimum information gain while spliting
  * @param row_rate sample ratio of train data set
  * @param col_rate sample ratio of features
  * @param num_trees number of decision trees
  */
private[tree] class RandomForest(impurity: String,
                                        max_depth: Int,
                                        max_bins: Int,
                                        bin_samples: Int,
                                        min_node_size: Int,
                                        min_info_gain: Double,
                                        row_rate: Double,
                                        col_rate: Double,
                                        num_trees: Int) extends Logging {

  /**
    * Method to train a random forest model over a training data which
    * epresented as an RDD of [[bda.common.obj.LabeledPoint]].
    *
    * @param train_data Training data points.
    * @return a [[bda.spark.model.tree.rf.RandomForestModel]] instance which can be used to predict.
    */
  def train(train_data: RDD[LabeledPoint]): RandomForestModel = {
    val timer = new Timer()

    val wk_learners = new ArrayBuffer[TreeNode]()

    var ind = 0
    while (ind < num_trees) {
      val wl = new CART(impurity,
        max_depth,
        max_bins,
        bin_samples,
        min_node_size,
        min_info_gain,
        row_rate,
        col_rate).train(train_data)
      wk_learners += wl.root

      logInfo(s"Random Forest Model tree#${ind + 1} training done")

      ind += 1
    }

    logInfo(s"Random Forest Model training done, cost time ${timer.cost()}ms")

    new RandomForestModel(wk_learners.toArray,
      impurity,
      max_depth,
      max_bins,
      bin_samples,
      min_node_size,
      min_info_gain,
      row_rate,
      col_rate,
      num_trees)
  }
}