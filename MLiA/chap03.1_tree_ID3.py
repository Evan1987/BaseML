
# coding: utf-8

import pandas as pd
import math

# 计算香农熵，输入集的最后一个值是label
def calcShannonEnt(dataSet):
    totalNum = len(dataSet)
    labels = pd.Series([x[-1] for x in dataSet])
    labelsSummary = labels.groupby(labels).size().apply(lambda x :(-x/totalNum)*math.log2(x/totalNum))
    result =labelsSummary.sum()
    return result
# 输入划分轴（特征）和匹配值后，对现有数据集进行切分
def splitDataSet(dataSet, axis, value):
    result = [x[:axis]+x[axis+1:] for x in dataSet if x[axis]==value]    
    return result
# 输入划分轴（特征）后，计算对现有数据集进行切分后的信息熵
def calcSplitMethodEnt(dataSet, axis):
    uniqueVals = set([x[axis] for x in dataSet])
    subSetList = [splitDataSet(dataSet, axis=axis, value=j) for j in uniqueVals]
    newEntList = [len(z)/len(dataSet) * calcShannonEnt(z) for z in subSetList]
    result = sum(newEntList)
    return result
# 计算最佳划分特征
def chooseBestFeatureToSplit(dataSet):
    baseEnt = calcShannonEnt(dataSet)
    numFeatures = len(dataSet[0])-1
    bestInfoGain = 0.0
    for i in range(numFeatures):
        newEnt = calcSplitMethodEnt(dataSet, axis=i)
        infoGain = newEnt - baseEnt
        if(infoGain<bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 对于叶子节点仍然具有多个label的情况，需要根据多数label进行表决
def majorityCnt(labelList):
    labelList = pd.Series(labelList)
    voteLabel = labelList.groupby(by=labelList).size().sort_values(ascending=False).index[0]
    return voteLabel

# 递归方法对数据集进行树划分
def createTree(dataSet,featureName):
    labelList = [x[-1] for x in dataSet]
    # 现有数据label完全相同
    if(labelList.count(labelList[0])==len(labelList)):
        return labelList[0]
    # 现有数据的全部特征已遍历
    if(len(dataSet[0])==1):
        return majorityCnt(labelList)
    # 根据最佳划分维度进行划分
    bestAxis = chooseBestFeatureToSplit(dataSet)
    bestAxisName = featureName[bestAxis]
    mytree = {bestAxisName:{}}
    del featureName[bestAxis]
    featureValues = [x[bestAxis] for x in dataSet]
    uniqueVals = set(featureValues)
    for i in uniqueVals:
        subFeatureNames = featureName[:]
        # 对于不同分支继续进行树划分
        mytree[bestAxisName][i] = createTree(splitDataSet(dataSet,bestAxis,i),subFeatureNames)
    return mytree

mydata = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]

createTree(dataSet=mydata, featureName=["no surfacing", "flippers"])

