#! /usr/bin/python

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from timeit import Timer
import log
import data
import sys
import os
import loss


def regressionTree(train_path, predict_path):
	log.log("INFO", "generate baseline of regression tree ...")

	X_train, y_train = data.loadLibSVMFile(train_path)
	
	rt_model = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 20)
	rt_model.fit(X_train, y_train)

	se = loss.SquaredError()
	X_predict, y_label = data.loadLibSVMFile(train_path)
	y_predict = rt_model.predict(X_predict)
	for i in range(0, len(y_predict)):
		se.add(y_predict[i], y_label[i])
	
	log.log("INFO", "regression tree baseline: train_RMSE = %f" % se.getRMSE())

	se = loss.SquaredError()
	X_predict, y_label = data.loadLibSVMFile(predict_path)
	y_predict = rt_model.predict(X_predict)
	for i in range(0, len(y_predict)):
		se.add(y_predict[i], y_label[i])
	
	log.log("INFO", "regression tree baseline: test_RMSE = %f" % se.getRMSE())

	return

if __name__ == "__main__":
	log.log("INFO", "main function of baseline")

	data_path = "/Users/hugh_627/ICT/bda/gboost/data/cadata"
	train_rate = 0.7

	data.sampleNoRepeat(data_path, train_rate)

	regressionTree(data_path + ".part1", data_path + ".part2")
