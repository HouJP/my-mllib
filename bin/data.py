#! /usr/bin/python

from sklearn.datasets import load_svmlight_file
import log
import sys
import os
import random

# Load data with libsvm format
def loadLibSVMFile(path):
	log.log("INFO", "load data from " + path + " ...")

	X_train, y_train = load_svmlight_file(path)

	log.log("INFO", "load data done.")

	return X_train, y_train

# Sampling without repeating for data file and generate part1-file and part2-file.
#	Num(part1-file) / Num(file) = rate
#	Num(part2-file) / Num(file) = 1 - rate
def sampleNoRepeat(path, rate):
	log.log("INFO", "sampling without repeating ...")

	f_all = []

	f = open(path)
	for line in f:
		line = line.strip()
		f_all.append(line)
	f.close()

	index = [i for i in range(len(f_all))]
	random.shuffle(index)

	n = len(f_all)
	n1 = int(rate * n)

	f = open(path + ".part1", 'w')
	for i in range(0, n1):
		f.write(f_all[index[i]] + "\n")
	f.close()

	f = open(path + ".part2", 'w')
	for i in range(n1, n):
		f.write(f_all[index[i]] + "\n")
	f.close()

	log.log("INFO", "sampling without repeating done.")

	return

if __name__ == "__main__":
	log.log("INFO", "main function of data")

	f = "/Users/hugh_627/ICT/bda/gboost/data/cadata"

	# test loadLibSVMFile function
	loadLibSVMFile(f)

	# test sampleNoRepeat function
	sampleNoRepeat(f, 0.7)
