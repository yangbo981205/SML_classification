"""
@File:test.py
@Date:2021/6/15 22:17
@Author:博0_oer~
"""
# import scipy.io as scio
# import numpy as np
#
# dataFile = "../StatisticalMachineLearning/benchmarks.mat"
# data = scio.loadmat(dataFile)
# print(data["titanic"])
# print(type(data))
# print(len(data))
#
# print(data["breast_cancer"])
# print(data)
# print(len(data))
# print(data.keys())

# print(data)
# for key in list(data.keys())[3:-1]:
#     print("dataset:{},len:{}".format(key, len(data[key])))
#     print(data[key])


# n = np.array([[(np.array([[1, 2], [1, 2], [1, 2]]), np.array([[1], [1], [-1]], dtype=np.int16),
#                np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]]), np.array([[7, 8, 9], [7, 8, 9], [7, 8, 9]]))]]
#              , dtype=[('x', 'O'), ('t', 'O'), ('test', 'O'), ('train', 'O')])
# print(n[0][0][1])
#
# print([[(4)]])

# print(len(data["banana"][0][0][0]))
# print(len(data["banana"][0][0][1]))
# print(len(data["banana"][0][0][2]))
# print(len(data["banana"][0][0][3]))

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.datasets import make_classification
# from sklearn.datasets import make_classification
#
# from sklearn.datasets import make_classification
#
# X, y = make_classification(n_samples=1000, n_features=2,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
#
# print(X,y)
#
# print(X, y)
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# print(clf.fit(X, y))
#
# print(clf.predict([[0, 0, 0, 0]]))
#
# print(clf.score(X, y))


# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import NuSVC
# clf = make_pipeline(StandardScaler(), NuSVC())
# clf.fit(X, y)
#
# print(clf.predict([[-0.8, -1]]))


# from sklearn.datasets import load_iris
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)
# iris = load_iris()
# print(iris.data, iris.target)
# print(cross_val_score(clf, iris.data, iris.target, cv=10))


# import sys
# #start
# output = sys.stdout
# outputfile = open("run_result.txt", 'w')
# sys.stdout = outputfile
#
# print("5465341561651")
# #end
# outputfile.close()
# sys.stdout = output

