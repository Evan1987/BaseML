
# coding: utf-8

import numpy as np
import random

# return matrix shape m*n：m-样本数，n-特征数
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return np.mat(dataMat)


# 用欧式距离计算向量间的距离
# return double
def distEclud(vecA, vecB):
    dist = np.sum(np.power(vecA-vecB, 2))**0.5
    return dist

# 随机生成簇质点
# return matrix shape k*n：k-质点数，n-特征数
def randCent(dataSet, k):
    # 数据集的维数
    n = np.shape(dataSet)[1]
    # 生成多组 k*n 的随机组合，便于之后生成簇质点
    # 用 set来当容器，方便保证组合没有重复
    rand = set()
    while len(rand) < k:
        rand.add(tuple([random.random() for _ in range(n)]))
    randMat = np.mat(list(map(list, list(rand))))
    # 各维度的统计性数据
    minMat = np.apply_along_axis(arr=dataSet, axis=0, func1d=np.min)
    maxMat = np.apply_along_axis(arr=dataSet, axis=0, func1d=np.max)
    rangeMat = maxMat - minMat
    
    centroids = minMat + np.multiply(randMat, rangeMat)
    return centroids

path = "F:/for learn/MLiA/Ch10/"
fileName = path + "testSet.txt"
dataSet = loadDataSet(fileName)


def kMeans(dataSet, k, distFun=distEclud, createCentFun=randCent):
    centroids = createCentFun(dataSet, k)
    totalSSE = changeSSE = np.inf    
    # 给出每个样本最近的簇质点序号以及 SE
    # return (index:Int, dist^2:Double)
    def getIndex(vecA, centroids, distFun):
        distEval = np.array([distFun(vecA, x) for x in centroids])
        index = distEval.argmin()
        dist = distEval[index]
        return index, dist**2
    
    # 根据生成的簇，计算新的簇质心
    # return matrix shape k*n: k-质点数，n-特征数
    def generateNewCentroids(dataSet, index, k):
        newCentroids = np.mat([np.apply_along_axis(
                               arr=dataSet[index == i],
                               axis=0,
                               func1d=np.mean) for i in range(k)])
        return newCentroids
        
    # 收敛条件：改变的 SSE≤1
    while changeSSE>1:
        clusterAssment = np.array([getIndex(x, centroids, distFun) for x in dataSet])
        index = clusterAssment[:, 0]
        sse = np.sum(clusterAssment[:, 1])
        changeSSE = totalSSE - sse
        print("now SSE is:", sse, "\t", "change SSE is:", changeSSE)
        totalSSE = sse
        centroids = generateNewCentroids(dataSet, index, k)
    
    return centroids, clusterAssment

result = kMeans(dataSet, k=5)

result[0]


result[1]

