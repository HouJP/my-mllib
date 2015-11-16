package bda.spark.ml.model

case class Split(feature: Int, threshold: Double) {

  override def toString: String = {
    s"feature = $feature, threshold = $threshold"
  }
}

class LowestSplit(feature: Int) extends Split(feature, Double.MinValue)

class HighestSplit(feature: Int) extends Split(feature, Double.MaxValue)