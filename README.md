****

# <center>Machine Learning Lib</center>
#### <center>Author: HouJP</center>
#### <center>E-mail: houjp1992@gmail.com</center>

****

### 目录
*	[项目介绍](#intro)
*	[使用说明](#usage)
*	[算法测试](#test)
*	[数据说明](#data)
*	[版本更新](#version)

****

###<a name="intro">项目介绍</a>

本项目基于Spark分布式计算框架，实现了目前流行的机器学习算法(分布式版本)。

当前包含的机器学习算法如下：

*	Classification And Regression Trees （used for classification and regression）
*	Gradient Boosting Decision Trees (used for multiple classification)
*	Gradient Boosting Regression Trees (used for regression)
*	Random Forest (used for regression)

设计文档请点击[这里](https://github.com/HouJP/my-mllib/blob/master/doc/trees.pdf)获取。

****

###<a name="usage">使用说明</a>

#### CART算法
*	分类	
	
```scala
// read training data and testing data from disk
val train = Points.readLibSVMFile(sc, data_dir + "a1a").cache()
val test = Points.readLibSVMFile(sc, data_dir + "a1a.t").cache()

// train a model of CART for classification
val cart_model = CART.train(
  train,
  impurity = "Gini",
  max_depth = 10,
  max_bins = 32,
  bin_samples = 10000,
  min_node_size = 15,
  min_info_gain = 1e-6)

// show structure of CART model
cart_model.printStructure()

// predict for testing data using the model
val preds = cart_model.predict(test)
// calculate testing error
val err = preds.filter(r => r._2 != r._3).count().toDouble / test.count()
println(s"Test Error: $err")
```

* 回归

```scala
// read training data and testing data from disk
val train = Points.readLibSVMFile(sc, data_dir + "cadata.train").cache()
val test = Points.readLibSVMFile(sc, data_dir + "cadata.test").cache()

// train a model of CART for regression
val cart_model = CART.train(
  train,
  impurity = "Variance",
  max_depth = 15,
  max_bins = 32,
  bin_samples = 10000,
  min_node_size = 10,
  min_info_gain = 1e-6)

// show structure of CART model
cart_model.printStructure()

// predict for testing data use the model
val preds = cart_model.predict(test)
// calculate testing error
println(s"Test RMSE: ${RMSE(preds.map(e => (e._2, e._3)))}")
\end{lstlisting}  
```

#### GBDT算法

```scala
// read training data and testing data from disk
val train = Points.readLibSVMFile(sc, data_dir + "cadata.train").cache()
val test = Points.readLibSVMFile(sc, data_dir + "cadata.test").cache()

// train a model of GBDT for multiple classification
val gbdt_model = GBDT.train(train,
  impurity = "Variance",
  max_depth = 15,
  max_bins = 32,
  bin_samples = 10000,
  min_node_size = 10,
  min_info_gain = 1e-6,
  num_round = 20)

// predict for testing data using the model
val preds = gbdt_model.predict(test)
// calculate testing error
val err = preds.filter(r => r._2 != r._3).count().toDouble / test.count()
println(s"Test Error: $err")
```

#### GBRT算法

```scala
// read training data and testing data from disk
val train = Points.readLibSVMFile(sc, data_dir + "cadata.train").cache()
val test = Points.readLibSVMFile(sc, data_dir + "cadata.test").cache()

// train a model of GBRT for regression
val gbrt_model = GBRT.train(
  train,
  Array(("test", test)),
  impurity = "Variance",
  max_depth = 15,
  max_bins = 32,
  bin_samples = 10000,
  min_node_size = 10,
  min_info_gain = 1e-6,
  num_round = 100,
  learn_rate = 0.1)

// predict for testing data using the model
val preds = gbrt_model.predict(test)
// calculate testing error
println(s”Test RMSE: ${RMSE(preds.map(e => (e._2, e._3)))}”)
```

#### RandomForest算法

*	分类

```scala
// read training data and testing data from disk
val train = Points.readLibSVMFile(sc, data_dir + "a1a").cache()
val test = Points.readLibSVMFile(sc, data_dir + "a1a.t").cache()

// train a model of Random Forest for classification
val rf_model = RandomForest.train(
  train,
  impurity = "Gini",
  max_depth = 10,
  max_bins = 32,
  bin_samples = 10000,
  min_node_size = 15,
  min_info_gain = 1e-6,
  row_rate = 0.6,
  col_rate = 0.6,
  num_trees = 100)

// predict for testing data using the model
val preds = rf_model.predict(test)
// calculate testing error
val err = preds.filter(r => r._2 != r._3).count().toDouble / test.count()
println(s"Test Error: $err")
```

*	回归

```scala
// read training data and testing data from disk
val train = Points.readLibSVMFile(sc, data_dir + "cadata.train").cache()
val test = Points.readLibSVMFile(sc, data_dir + "cadata.test").cache()

// train a model of Random Forest for regression
val rf_model = RandomForest.train(
  train,
  impurity = "Variance",
  max_depth = 15,
  max_bins = 32,
  bin_samples = 10000,
  min_node_size = 10,
  min_info_gain = 1e-6,
  row_rate = 0.6,
  col_rate = 0.6,
  num_trees = 100)

// predict for testing data use the model
val preds = rf_model.predict(test)
// calculate testing error
println(s"Test RMSE: ${RMSE(preds.map(e => (e._2, e._3)))}")
```


****

###<a name="test">算法测试</a>

* 测试环境
	* 集群大小: 1 * Master + 5 * Worker
	* 操作系统: CentOS release 6.6 (Final)
	* 单机内存: 132G
	* 单机硬盘: 1.8T
	
#### CART算法

| 算法 | 数据集 | 训练集评测 | 测试集评测 | 时间 | 参数 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| tree.cart.CART | cadata(3:1) | RMSE(57515.27) | RMSE(60845.42) | 9128ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6) |
| tree.cart.CART | cadata(3:1) | RMSE(52695.06) | RMSE(59125.73) | 9128ms | impurity(Variance),max_depth(15),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6) |
| tree.cart.CART | a1a/a1a.t | Acc(0.8274),Pre(0.6453),Rec(0.6633),Auc(0.7721) | Acc(0.8189),Pre(0.6201),Rec(0.6377),Auc(0.7570) | 6163ms | impurity(Gini),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6) |
| tree.cart.CART | a1a/a1a.t | Acc(0.8399),Pre(0.6994),Rec(0.6127),Auc(0.7634) | Acc(0.8167),Pre(0.6368),Rec(0.5539),Auc(0.7269) | 6163ms | impurity(Gini),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6) |


#### GBDT算法

| 算法 | 数据集 | 训练集评测 | 测试集评测 | 时间 | 参数 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| tree.gbdt.GBDT | a1a/a1a.t| Acc(0.8411),Pre(0.7414),Rec(0.5443),Auc(0.7412) | Acc(0.8190),Pre(0.6551),Rec(0.5231),Auc(0.7179)  | 7458ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),num_round(1) |
| tree.gbdt.GBDT | a1a/a1a.t| Acc(0.8717),Pre(0.7981),Rec(0.6405),Auc(0.7938) | Acc(0.8209,Pre(0.6481),Rec(0.5586),Auc(0.7313)  | 10262ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),num_round(2) |
| tree.gbdt.GBDT | a1a/a1a.t| Acc(0.8903),Pre(0.8174),Rec(0.7139),Auc(0.8309) | Acc(0.8226),Pre(0.6385),Rec(0.6050),Auc(0.7483)  | 13023ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),num_round(3) |
| tree.gbdt.GBDT | a1a/a1a.t| Acc(0.9539),Pre(0.9147),Rec(0.8962),Auc(0.9345) | Acc(0.8135),Pre(0.6197),Rec(0.5811),Auc(0.7341)  | 25707ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),num_round(10) |


#### GBRT算法


| 算法 | 数据集 | 训练集评测 | 测试集评测 | 时间 | 参数 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| tree.gbrt.GBRT | cadata(3:1) | RMSE(53397.68) | RMSE(60305.92) | 35431ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),num_round(10),learn_rate(0.02) |
| tree.gbrt.GBRT | cadata(3:1) | RMSE(45591.16) | RMSE(57616.83) | 52898ms | impurity(Variance),max_depth(15),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6),num_round(10),learn_rate(0.02) |
| tree.gbrt.GBRT | cadata(3:1) | RMSE(38681.74) | RMSE(56180.71) | 434537ms | impurity(Variance),max_depth(15),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6),num_round(50),learn_rate(0.02) |
| tree.gbrt.GBRT | cadata(3:1) | RMSE(33826.15) | RMSE(55246.31) | 1057077ms | impurity(Variance),max_depth(15),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6),num_round(100),learn_rate(0.02) |

#### RandomForest算法

| 算法 | 数据集 | 训练集评测 | 测试集评测 | 时间 | 参数 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| tree.rf.RandomForest | cadata(3:1) | RMSE(53682.94) | RMSE(56607.57) | 42313ms | impurity(Variance),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),row_rate(0.6),col_rate(0.6),num_trees(20) |
| tree.rf.RandomForest | cadata(3:1) | RMSE(48106.61) | RMSE(53945.51) | 58230ms | impurity(Variance),max_depth(15),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6),row_rate(0.6),col_rate(0.6),num_trees(20) |
| tree.rf.RandomForest | a1a/a1a.t | Acc(0.8436),Pre(0.7209),Rec(0.5949),Auc(0.7599) | Acc(0.8281),Pre(0.6714),Rec(0.5590),Auc(0.7362) | 40484ms | impurity(Gini),max_depth(10),max_bins(32),bin_samples(10000),min_node_size(15),min_info_gain(1e-6),row_rate(0.6),col_rate(0.6),num_trees(20) |
| tree.rf.RandomForest | a1a/a1a.t | Acc(0.8530),Pre(0.7331),Rec(0.6329),Auc(0.7789) | Acc(0.8271),Pre(0.6594),Rec(0.5818),Auc(0.7433) | 44227ms | impurity(Gini),max_depth(15),max_bins(32),bin_samples(10000),min_node_size(10),min_info_gain(1e-6),row_rate(0.6),col_rate(0.6),num_trees(20) |


****

###<a name="data">数据说明</a>

| Name | Type | Size | \# of classes | \# of data | \# of features | Directory | Source |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |  
| cadata | regression | 4.6M | - | 20,640 | 8 | project_dir/data/regression | [Link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cadata) |
| a1a/a1a.t | classification | 112K/2.1M | 2| 2,265/30,296 | 123 | project_dir/data/classification | [Link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a) |

****

###<a name="version">Version Updating</a>
*	04/15/2016
	*	修改weighted——impurity的计算方式：如果左/右孩子样本点个数小于节点最小尺寸，则将WeightedImpurity置为MaxDouble，防止左/右孩子样本点个数小于节点最小尺寸且WeightedImpurity小于MinInfoGain时，阻碍节点分裂。

*	04/11/2016
	*	完成Spark版本的GBDT(Gradient Boosting Decision Trees)算法，用于多分类。

*	03/09/2016
	*	完成CART(Classification And Regression Trees)，可以用作分类(Gini)和回归(Variance)。
	
*	12/31/2015
	*	更改单机决策树的实现方法(使用分箱)
	*	实现随机森林

*	12/01/2015
	*	改变分布式版本决策树的实现方法(使用分箱)
	*	实现bda.spark.model.tree.DecisionTree and bda.spark.model.tree.GradientBoost

*	10/15/2015
	*	修正StackOverflowError(当迭代次数>300时)
	*	测试bda.spark.ml.GBoost
		
*	10/14/2015
	*	实现GBDT的spark版本
	*	问题
		*	当迭代次数很大时，报StackOverflowError
		
*	09/29/2015
	*	使用xgboost作为baseline

*	09/27/2015
	*	实现单机版本GBDT

*	09/25/2015
	*	使用sklearn.tree.DecisionTreeRegressor作为baseline

*	09/23/2015
	*	实现单机版本决策树(回归)

****
