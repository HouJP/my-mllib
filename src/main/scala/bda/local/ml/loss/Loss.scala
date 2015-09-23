package bda.local.ml.loss

import bda.local.ml.model.Stat

trait Loss {

  def gradient(prediction: Double, label: Double): Double

  def computeError(prediction: Double, label: Double): Double

  def predict(stat: Stat): Double
}