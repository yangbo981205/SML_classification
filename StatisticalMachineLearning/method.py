"""
@File:method.py
@Date:2021/6/16 16:31
@Author:博0_oer~
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier


def Adaboost(train_data, train_lable, test_data, test_lable):
    # 构造分类器
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    # 从训练集构造一个提升分类器
    clf.fit(train_data, train_lable)
    # 使用分类器进行预测打分
    return clf.score(test_data, test_lable)


# 岭回归和分类
def Ridge(train_data, train_lable, test_data, test_lable):
    clf = RidgeClassifier().fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 线性判别分析
def LinearDiscriminant(train_data, train_lable, test_data, test_lable):
    clf = LinearDiscriminantAnalysis().fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 支持向量机
# SVM-Svc
def Svc(train_data, train_lable, test_data, test_lable):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# SVM-NuSvc
def NuSvc(train_data, train_lable, test_data, test_lable):
    clf = make_pipeline(StandardScaler(), NuSVC())
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 随机梯度下降SDG
def SGD(train_data, train_lable, test_data, test_lable):
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3))
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 最近邻
# KNeighborsClassifier
def KNeighbors(train_data, train_lable, test_data, test_lable):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# NearestCentroid
def Nearest(train_data, train_lable, test_data, test_lable):
    clf = NearestCentroid()
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 高斯过程
def GaussianProcess(train_data, train_lable, test_data, test_lable):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 决策树
def DecisionTree(train_data, train_lable, test_data, test_lable):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)


# 概率校准
def Calibrated(train_data, train_lable, test_data, test_lable):
    base_clf = GaussianNB()
    calibrated_clf = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
    calibrated_clf.fit(train_data, train_lable)
    return calibrated_clf.score(test_data, test_lable)


# 神经网络
def MLP(train_data, train_lable, test_data, test_lable):
    clf = MLPClassifier(random_state=1, max_iter=10000)
    clf.fit(train_data, train_lable)
    return clf.score(test_data, test_lable)

