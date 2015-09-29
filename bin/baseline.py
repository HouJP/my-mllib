#! /usr/bin/python

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import scipy.sparse
import xgboost as xgb

from timeit import Timer
import time
import log
import data
import sys
import os
import loss

def gBoost(train_path, test_path):
	log.log("INFO", "generate baseline for gboost ...")

	d_train = xgb.DMatrix(train_path)
	d_test = xgb.DMatrix(test_path)

	para = {'booster': 'gbtree', 'silent': 1, 'nthread': 10, 'eta': 0.01, 'max_depth': 20, 'min_child_weight': 20}
	num_round = 400

	t_start = time.clock()
	gb_model = xgb.train(para, d_train, num_round)
	t_end = time.clock()
	log.log("INFO", "training model cost time: %d" % (t_end - t_start))

	# train RMSE
	y_predict = gb_model.predict(d_train)
	y_label = d_train.get_label()
	se = loss.SquaredError()
	for i in range(0, len(y_predict)):
		se.add(y_predict[i], y_label[i])
	log.log("INFO", "gboost baseline: train_RMSE = %f" % se.getMean())

	# test RMSE
	y_predict = gb_model.predict(d_test)
	y_label = d_test.get_label()
	se = loss.SquaredError()
	for i in range(0, len(y_predict)):
		se.add(y_predict[i], y_label[i])
	log.log("INFO", "gboost baseline: test_RMSE = %f" % se.getMean())

	return

def regressionTree(train_path, predict_path):
	log.log("INFO", "generate baseline for regression tree ...")

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

	# regressionTree(data_path + ".part1", data_path + ".part2")

	gBoost(data_path + ".part1", data_path + ".part2")
