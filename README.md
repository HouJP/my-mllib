****

#<center>GBoost</center>
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

GBoost implements the gradient boosting and some other algorithms such as decision tree used as weak learner. They can be used together to form the additive model such as GBDT and so on.

****

###<a name="usage">Direction for Use</a>

TODO

****

###<a name="test">Algorithm Testing</a>

#### GBoost

* Enviroment
	* CPU: 1.3 GHz Intel Core i5
	* Memory: 4 GB 1600 MHz DDR3
	
* DataSet
	* Name: cadata
	* Rate: 70%-training, 30%-testing
	
* Results Table:
	* xgboost
		* Fixed-parameters: gamma = 0, max_delata_step = 0, subsample = 1, colsample_byree = 1, lambda = 1, alpha = 0, nthread = 10
	* GBoost
		* Fixed-parameters: none
		
| Algorithm | num_iter | eta | max_depth | min_child_weight | total_time | train_RMSE | test_RMSE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| xgboost 	| 100 | 0.01 | 15 | 10 | 4s | 97536.96 | 102027.12 |
| 			| 200 | 0.01 | 15 | 10 | 9s | 48573.12 | 58397.00 |
| 			| 300 | 0.01 | 15 | 10 | 14s | 31399.54 | 49394.50 |
| 			| 400 | 0.01 | 15 | 10 | 19s | 24620.25 | 48155.99 |
| 			| 500 | 0.01 | 15 | 10 | 23s | 23296.80 | 47144.61 |
| 			| 600 | 0.01 | 15 | 10 | 26s | 21310.44 | 48077.48 |
| 			| 600 | 0.01 | 15 | 20 | 24s | 28383.57 | 47840.43 |
| 			| 400 | 0.01 | 20 | 20 | 21s | 30126.81 | 48200.24 |
| 			| 500 | 0.01 | 20 | 20 | 26s | 28251.16 | **46656.16** |
| 			| 600 | 0.01 | 20 | 20 | 31s | 27059.08 | 46704.55 |
|			| 700 | 0.01 | 20 | 20 | 35s | 25854.79 | 46895.51 |
| 			| 600 | 0.01 | 21 | 20 | 27s | 27478.99 | 47756.24 |

| Algorithm | num_iter | learn_rate | max_depth | min_node_size | ave_iter_cost_time | train_RMSE | test_RMSE |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| bda.local.ml.GBoost 	| 100 | 0.01 | 15 | 10 | 857ms | 46993.31 | 60005.90 |
| 					 	| 200 | 0.01 | 15 | 10 | 757ms | 25740.15 | 49021.34 |
| 					 	| 300 | 0.01 | 15 | 10 | 692ms | 20494.19 | 47727.48 |
| 					 	| 400 | 0.01 | 15 | 10 | 871ms | 18215.08 | 47458.97 |
| 					 	| 500 | 0.01 | 15 | 10 | 761ms | 16767.86 | 47326.49 |
|						| 1000 | 0.01 | 15 | 10 | 798ms | 12343.36 | **47144.03** |
|						| 100 | 0.01 | 15 | 20 | 913ms | 36081.40 | 60252.03 |
|						| 100 | 0.01 | 20 | 20 | 1144ms | 37641.07 | 59462.60 |


#### Decision Tree

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
| bda.local.ml.DTree 	| 15 | 10 | - | 32438.77 | 65735.05 |
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
*	2015/10/14
	*	implement gbdt algorithm running on spark
	*	issues
		*	meet StackOverflowError when number of iteration is too large.
*	2015/09/29
	*	use xgboost to generate baseline for performance evaluation

*	2015/09/27
	* 	implement gbdt algorithm running on local

*	2015/09/25
	*	use sklearn.tree.DecisionTreeRegressor to generate baseline for performance evaluation
	*	add code comments for decision tree

*	2015/09/23
	*	implementations
		*	implement decision tree which can be used to rain and predict and running on local
	*	issues
		* 	only for regression
		*	model can not be saved	

****
