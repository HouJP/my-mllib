****

#<center>my-mllib</center>
####<center>E-mail: houjp1992@gmail.com</center>

****

### Contents
*	[Project Introduction](#intro)
*	[Direction for Use](#usage)
*	[Algorithm Testing](#test)
*	[Data Specification](#data)
*	[Version Updating](#version)

****

###<a name="intro">Project Introduction</a>


The project implemented some machine learning algorithms on spark which is written in scala and it also included standalone implementations of these algorithms.

You can find these ML algorithms up to now:

*	Classification And Regression Trees （used for classification and regression）
*	Gradient Boosting Decision Trees (used for multiple classification)
*	Gradient Boosting Regression Trees (used for regression)
*	Random Forest (used for regression)

****

###<a name="usage">Direction for Use</a>

TODO

****

###<a name="test">Algorithm Testing</a>

#### bda.local.model.tree.GradientBoost

* Enviroment
	* CPU: 1.3 GHz Intel Core i5
	* Memory: 4 GB 1600 MHz DDR3
	
* DataSet
	* Name: cadata
	* Rate: 70%-training, 30%-testing
	
* Fixed parameters:
	* xgboost: gamma = 0, max_delata_step = 0, subsample = 1, colsample_byree = 1, lambda = 1, alpha = 0, nthread = 10
	* GradientBoost: data_sample_rate = 0.2, max_data_sample = 10000
		
| Algorithm | num_iter | eta | max_depth | min_child_weight | total_time | train_RMSE | test_RMSE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| xgboost 	| 100 | 0.01 | 15 | 10 | 4s | 97536.96 | 102027.12 |
| 			| 300 | 0.01 | 15 | 10 | 14s | 31399.54 | 49394.50 |
| 			| 500 | 0.01 | 15 | 10 | 23s | 23296.80 | 47144.61 |
| 			| 500 | 0.01 | 20 | 20 | 26s | 28251.16 | **46656.16** |
| 			| 600 | 0.01 | 20 | 20 | 31s | 27059.08 | 46704.55 |

| Algorithm | num_iter | learn_rate | max_depth | min_node_size | ave_iter_cost_time | train_RMSE | test_RMSE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GradientBoost 	| 100 | 0.01 | 15 | 10 | 857ms | 46993.31 | 60005.90 |
| 					 	| 500 | 0.01 | 15 | 10 | 761ms | 16767.86 | 47326.49 |
|						| 1000 | 0.01 | 15 | 10 | 798ms | 12343.36 | **47144.03** |
|						| 100 | 0.01 | 15 | 20 | 913ms | 36081.40 | 60252.03 |
|						| 100 | 0.01 | 20 | 20 | 1144ms | 37641.07 | 59462.60 |

#### bda.local.model.tree.DecisionTree

* Enviroment
	* CPU: 1.3 GHz Intel Core i5
	* Memory: 4 GB 1600 MHz DDR3
	
* DataSet
	* Name: cadata
	* Rate: 70%-training, 30%-testing

| Algorithm | max_depth | min_samples_leaf | cost_time | train_RMSE | test_RMSE |
| ---- | ---- | ---- | ---- | ---- | ---- |
| sklearn.tree.DecisionTreeRegressor 	| 15 | 10 | - | 45111.32 | 61751.50 |
|										| 15 | 20 | - | 50701.13 | 58233.37 |
|										| 20 | 20 | _ | 50758.53 | 59627.13 |
| | | | | | |
| DecisionTree 	| 15 | 10 | - | 32438.77 | 65735.05 |
|						| 15 | 20 | - | 39596.32 | 62751.22 |
|						| 20 | 20 | - | 37900.60 | 63104.04 |


****

###<a name="data">Data Specification</a>

#### cadata
*	Type: regression
*	Source: [StatLib/houses.zip](http://lib.stat.cmu.edu/datasets/houses.zip)
*	Homepage: [StatLib](http://lib.stat.cmu.edu/datasets/)
*	Size: 4.6M
*	\# of data: 20,640
*	\# of features: 8
*	Directory: $project/data/cadata

****

###<a name="version">Version Updating</a>

*	04/11/2016
	*	finished GBDT(Gradient Boosting Decision Trees) on Spark, used for multiple classification.

*	03/09/2016
	*	finished CART(Classification And Regression Trees), used gini for classification and variance for regression.
*	12/31/2015
	* 	changed implementation of standalone decision tree algorithm (use bins) which made tree algorithms run faster.
	*	implemented random forest algorithm of standalone/distributed mode.

*	12/01/2015
	*	changed implementation in distributed decision tree algorithm (use bins).
	*	implemented bda.spark.model.tree.DecisionTree and bda.spark.model.tree.GradientBoost.

*	10/15/2015
	*	fixed StackOverflowError when number of iteration is too large (>300).
	*	evaluate performance of bda.spark.ml.GBoost.
	
*	10/14/2015
	*	implement gbdt algorithm running on spark
	*	issues
		*	meet StackOverflowError when number of iteration is too large.
		
*	09/29/2015
	*	use xgboost to generate baseline for performance evaluation

*	09/27/2015
	* 	implement gbdt algorithm running on local

*	09/25/2015
	*	use sklearn.tree.DecisionTreeRegressor to generate baseline for performance evaluation
	*	add code comments for decision tree

*	09/23/2015
	*	implementations
		*	implement decision tree which can be used to rain and predict and running on local
	*	issues
		* 	only for regression
		*	model can not be saved	

****
