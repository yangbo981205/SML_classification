"""
@File:run.py
@Date:2021/6/16 12:04
@Author:博0_oer~
"""
from loaddata import loaddata, description, divide
from method import *
import logging
import sys


output = sys.stdout
outputfile = open("run_result.txt", 'w')
sys.stdout = outputfile
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.DEBUG)


if __name__ == '__main__':
    logging.info("Dataset is loading...... ")
    dataFile = r"../StatisticalMachineLearning/benchmarks.mat"
    data, lable = loaddata(dataFile)
    train_data, train_lable, test_data, test_lable = divide(data, lable)
    description(data, lable)

    logging.info("Dataset loading is complete.")
    print("Classification accuracy")
    for key in data.keys():
        logging.info("----------Classification of dataset " + key + "----------")
        print("------------------------------------------------------------------------------------")
        print("Classification results of various algorithms on dataset {}".format(key))
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("Adaboost", key, Adaboost(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("Adaboost is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("Ridge", key, Ridge(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("Ridge is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("LinearDiscriminant", key, LinearDiscriminant(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("LinearDiscriminant is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("Svc", key, Svc(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("Svc is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("NuSvc", key, NuSvc(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("NuSvc is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("SGD", key, SGD(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("SGD is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("KNeighbors", key, KNeighbors(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("KNeighbors is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("Nearest", key, Nearest(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("Nearest is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("GaussianProcess", key, GaussianProcess(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("GaussianProcess is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("DecisionTree", key, DecisionTree(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("DecisionTree is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("Calibrated", key, Calibrated(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("Calibrated is complete.")
        print("The accuracy of using {} method to classify {} is:{}%"
              .format("MLP", key, MLP(train_data[key], train_lable[key], test_data[key], test_lable[key])))
        logging.info("MLP is complete.")


outputfile.close()
sys.stdout = output



