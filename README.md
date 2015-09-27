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

#### Decision Tree

* DataSet: cadata
* Rate: 70%-training, 30%-testing

| Algorithm | max_depth | min_samples_leaf | cost_time | train_RMSE | test_RMSE |
| - | - | - | - | - | - |
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
*	2015/09/27
	* implement gbdt algorithm

*	2015/09/25
	*	use sklearn.tree.DecisionTreeRegressor to generate baseline for performance evaluation
	*	add code comments for decision tree

*	2015/09/23
	*	implementations
		*	implement decision tree which can be used to rain and predict
	*	issues
		* 	only for regression
		*	model can not be saved	

****
