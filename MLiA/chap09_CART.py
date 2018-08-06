
# coding: utf-8
import numpy as np
import pandas as pd

filePath = "F:/for learn/MLiA/Ch09/"
# 默认最后一列是输出值，前面的列都是特征
def loadDataSet(filePath):
    df = pd.read_table(filePath, header=None)
    dataMat = np.mat(df)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mask = (dataSet[:, feature] > value).A.flatten()
    mat0 = dataSet[mask, :]
    mat1 = dataSet[~mask, :]
    return mat0, mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # tolN:最小分割样本数
    # tolS：目标函数最小优化值（目标分数减少值）
    tolS, tolN = ops
    if(len(set(dataSet[:, -1].A.flatten()))==1):
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    # 遍历所有特征
    for featIndex in range(n-1):
        targetSet = set(dataSet[:, featIndex].A.flatten())
        # 遍历所有值（每个值都有可能是一个分割节点）
        for splitVal in targetSet:
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if((np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN)):
                continue
            newS = errType(mat0) + errType(mat1)
            if(newS<bestS):
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS<tolS):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if(feat==None):
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


#----剪枝部分----
## 判断是否为树结构，true：非叶子节点； false：叶子节点。
def isTree(obj):
    return type(obj).__name__ == 'dict'
## 递归塌陷式求叶子节点平均值
def getMean(tree):
    if isTree(tree["right"]):
        tree["right"] = getMean(tree["right"])
    if isTree(tree["left"]):
        tree["left"] = getMean(tree["left"])
    return 0.5*(tree["left"]+tree["right"])

def prune(tree,testData):
    # 如果测试集为空，则直接塌陷树
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    ########## prune给每个节点返回树或者值 ###################
    # cond 1
    # 如果左右部分至少有一棵是树，则在此分裂点对测试集进行分裂，得到左右两个测试集
    if isTree(tree["right"]) or isTree(tree["left"]):
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
    # --cond 1.1
    # --如果左边是树，那么利用测试集的分裂左集进行递归剪枝
    if isTree(tree["left"]):
        tree["left"] = prune(tree["left"], lSet)
    # --cond 1.2
    # --如果右边是树，那么利用测试集的分裂右集进行递归剪枝
    if isTree(tree["right"]):
        tree["right"] = prune(tree["right"], rSet)
    
    # cond 2
    # 如果两边都不是树（都是叶子节点），那么需要对其进行剪枝判断
    if not isTree(tree["right"]) and not isTree(tree["left"]):
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
        # 如果不剪枝（不合并），则体现在测试集的差距为：
        errorNoMerge = sum(np.power(lSet[:, -1]-tree["left"], 2)) + sum(np.power(rSet[:, -1]-tree["right"], 2))
        treeMean = 0.5*(tree["right"]+tree["left"])
        # 如果剪枝（合并），则体现在测试集上的差距为：
        errorMerge = sum(np.power(testData[:, -1]-treeMean, 2))
        # 如果剪枝要更好（差距小），则进行剪枝处理
        if errorMerge<errorNoMerge:
            print("Merging")
            return treeMean
        else:
            return tree
    else:
        print("missing comparation")
        return tree

# 测试剪枝算法

myDat = loadDataSet(filePath+"ex2.txt")
myDatTest = loadDataSet(filePath+"ex2test.txt")


myTree = createTree(myDat, ops=(0, 1))
myTree

prunedMyTree = prune(myTree, myDatTest)
prunedMyTree


dataSet = loadDataSet(filePath+"exp2.txt")

# ----------------------------- Model Tree -----------------------------------
def linearSolve(dataSet):
    dataSet = np.mat(dataSet)
    m, n = dataSet.shape
    X = np.mat(np.ones((m, n)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        raise NameError("This matrix is singular, cannot do inverse!Try increase the second par of ops!")
    w = xTx.I*(X.T*Y)
    return w, X, Y

def modelLeaf(dataSet):
    w, X, Y = linearSolve(dataSet)
    return w

def modelErr(dataSet):
    w, X, Y = linearSolve(dataSet)
    # yHat为一个列向量，对于其中每一个y值 = w.T*(X[i,:].T)
    # 在代码中X每一样例都是行向量
    yHat = X*w
    return sum(np.power(yHat-Y, 2))

createTree(dataSet, leafType=modelLeaf, errType=modelErr, ops=(1, 10))

createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 10))

