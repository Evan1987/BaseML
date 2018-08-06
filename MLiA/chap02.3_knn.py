
# coding: utf-8
import pandas as pd
import numpy as np
import os

# 读取txt文档，并将其变为 1维 array
def file2array(filePath):
    rawStr = np.array(open(filePath).readlines())
    strArray = np.array([list(x.replace("\n","")) for x in rawStr]).ravel().astype(int)
    return strArray

# 汇总训练集文件，并形成训练矩阵
def getTrainSet(trainPath):
    trainFiles = os.listdir(trainPath)
    trainNum = len(trainFiles)
    labels = [x.split("_")[0] for x in trainFiles]
    trainSet = [file2array(trainPath+x) for x in trainFiles]
    return labels,trainSet

# 单一预测函数，对任意相同格式的待预测数据，进行KNN预测
def classify0(testFilePath,trainLabels,trainSet,k=3):
    testData = file2array(testFilePath)
    trainLabels = pd.Series(trainLabels)
    f = lambda x: round(((x-testData)**2).sum()**0.5,2)
    selectDist = pd.Series([f(x) for x in trainSet]).sort_values(ascending=True)[:k]
    selectLabel = trainLabels[trainLabels.index.isin(selectDist.index)]
    label = selectLabel.groupby(by=selectLabel).size().sort_values(ascending=False).index[0]
    return label

# 批量预测函数，对批量待预测文件进行预测，并输出实际分类与预测分类
def batClassifyTest(testPath,trainLabels,trainSet,k=3):
    testFiles = os.listdir(testPath)
    exp_labels = [x.split("_")[0] for x in testFiles]
    pred_labels = [classify0(testPath+x,
                             trainLabels=trainLabels,
                             trainSet=trainSet,
                             k=k) for x in testFiles]
    df = pd.DataFrame({"exp_label":exp_labels,"pred_label":pred_labels})
    return df

path = "F:/for learn/MLiA/Ch02/"
trainPath = path+"trainingDigits/"
testPath = path+"testDigits/"
trainResult = getTrainSet(trainPath)
trainLabels = trainResult[0]
trainSet = trainResult[1]

result = batClassifyTest(testPath,trainLabels,trainSet,k=3)

errorRatio = round(len(np.where(result.exp_label!=result.pred_label)[0])/result.shape[0],4)
errorRatio

