package bda.example.a1a

import bda.example._
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object RunMLLibDecisionTree {

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(s"MLLib Decision Tree Demo of a1a dataset")
      .set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)

    val dir = input_dir + "classification/a1a/"
    val train_pt = dir + "a1a.nnl"
    val test_pt = dir + "a1a.t.nnl"
    val model_pt = output_dir + "dt.model"

    // Load and parse the data file.
    val trainingData = MLUtils.loadLibSVMFile(sc, train_pt)
    val testData = MLUtils.loadLibSVMFile(sc, test_pt)

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 10
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

    // Save and load model
    //model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
    //val sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
  }
}