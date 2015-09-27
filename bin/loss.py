#! /usr/bin/python

import math
import log

class SquaredError(object):
	
	def __init__(self):
		self.__sumSE = 0.0
		self.__count = 0

	def add(self, pre, label):
		self.__sumSE += SquaredError.computeError(pre, label)
		self.__count += 1
		return

	def getRMSE(self):
		return math.sqrt(self.__sumSE / self.__count)

	@staticmethod
	def computeError(pre, label):
		err = pre - label
		return err * err

if __name__ == "__main__":
	log.log("INFO", "main function of loss")

	# test SquaredError.computeError
	pre = 0.5
	label = 0.7
	se = SquaredError.computeError(pre, label)
	print "pre = %f, label = %f, squared error = %f" % (pre, label, se)

	# test SquaredError.getRMSE
	se = SquaredError()
	se.add(0.5, 0.7)
	se.add(0.2, 0.9)
	print "rmse = %f" % se.getRMSE()
