# SML_classification
Homework of Statistical Machine Learning.

## 描述

调用了`sklearn`包里面的分类相关算法，对分类领域13个`benchmarks`数据集进行分类任务的训练。

**loaddata.py：**

进行数据集的加载和处理，使得数据可以适应算法输入的要求，并将数据集中的信息进行统计输出

训练集和测试集设置：

因为数据集中数据并没有规律，所以直接选取数据中的前80%数据作为训练集，后30%数据作为测试集，训练集和测试集有10%的重复数据

**method.py：**

调用了sklearn中的常用分类算法，并设置了必要的参数。

算法分别为：

- AdaBoostClassifier
- RidgeClassifier（线性模型）
- LinearDiscriminant（线性和二次判别模型）
- SVC（支持向量机）
- NuSVC（支持向量机）
- SGDClassifier（随机梯度下降）
- KNeighborsClassifier（最近邻）
- NearestCentroid（最近邻）
- GaussianProcessClassifier（高斯过程）
- DecisionTreeClassifier（决策树）
- CalibratedClassifierCV（概率校准分类器）
- MLPClassifier（多层感知机神经网络模型）

**run.py：**

程序运行入口，输出运行日志，并将运行结果存入文件

输出内容：各种算法在测试集上的准确率

