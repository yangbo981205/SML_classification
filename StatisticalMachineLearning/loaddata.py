"""
@File:loaddata.py
@Date:2021/6/16 11:28
@Author:博0_oer~
"""
import scipy.io as scio
import numpy as np
import sys


output = sys.stdout
outputfile = open("run_result.txt", 'w')
sys.stdout = outputfile


def loaddata(datafile):
    """
    加载数据集
    :param datafile:
    :return:
    """
    data = scio.loadmat(datafile)
    datas = {}
    lable = {}
    for key in list(data.keys())[3:-1]:
        datas[key] = data[key][0][0][0]
        lable[key] = data[key][0][0][1]
    return datas, lable



def description(data_, lable_):
    """
    数据集描述
    :param data_:
    :param lable_:
    :return:
    """
    print("-----------------------------------------------------------------------------------------------------------")
    print("Dataset Description:")
    for key in data_.keys():
        print("dataset_name:%-15s  data_number:%-5d  data_dimension:%-5d  positive_sample:%-5d  negative_sample:%-5d"
              % (key, len(data_[key]), len(data_[key][0]), list(lable_[key]).count(1), list(lable_[key]).count(-1)))
    print("-----------------------------------------------------------------------------------------------------------")


def divide(data_, lable_):
    """
    进行训练集和测试集划分
    规则：因为数据集并没有规律，所以直接取前80%做训练集，后30%做测试集
    :param data_:
    :param lable_:
    :return:
    """
    train_data = {}
    test_data = {}
    train_lable = {}
    test_lable = {}
    for key in data_.keys():
        train_data[key] = data_[key][0:int(len(data_[key])*0.8)]
        test_data[key] = data_[key][int(-len(data_[key])*0.3):]
        train_lable[key] = lable_[key][0:int(len(data_[key])*0.8)]
        test_lable[key] = lable_[key][int(-len(data_[key])*0.3):]

        train_lable[key] = [i for t in train_lable[key] for i in t]
        test_lable[key] = [i for t in test_lable[key] for i in t]

    return train_data, train_lable, test_data, test_lable


outputfile.close()
sys.stdout = output


if __name__ == '__main__':
    dataFile = r"../StatisticalMachineLearning/benchmarks.mat"
    data, lable = loaddata(dataFile)
    train_data, train_lable, test_data, test_lable = divide(data, lable)
    # description(data, lable)



