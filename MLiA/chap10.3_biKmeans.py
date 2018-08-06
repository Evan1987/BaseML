
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
    while changeSSE > 1:
        clusterAssment = np.mat([getIndex(x, centroids, distFun) for x in dataSet])
        index = clusterAssment[:, 0].A1
        sse = np.sum(clusterAssment[:, 1])
        changeSSE = totalSSE - sse
        print("now SSE is:", sse, "\t", "change SSE is:", changeSSE)
        totalSSE = sse
        centroids = generateNewCentroids(dataSet, index, k)
    
    return centroids, clusterAssment

path = "F:/for learn/MLiA/Ch10/"
fileName = path + "testSet.txt"
dataSet = loadDataSet(fileName)

result = kMeans(dataSet, k=5)

# 二分K-均值聚类算法
def biKmeans(dataSet, k, distFun=distEclud):
    m = dataSet.shape[0]
    # 最开始将所有数据看成是一个簇，初始化簇类别（0）和距离矩阵 clusterAssment
    centroids = np.mean(dataSet, axis=0)
    clusterAssment = np.mat([(0, distFun(x, centroids)) for x in dataSet])
    while centroids.shape[0] < k:
        lowestSSE = np.inf
        # 遍历当前所有簇，比较从任意簇进行 2元分裂减少的 SSE
        for i in range(centroids.shape[0]):
            ptsInCurrCluster = dataSet[clusterAssment[:, 0].A1 == i]
            newCentroids, splitClustAss = kMeans(ptsInCurrCluster, 2, distFun)
            # 不属于该簇的 SSE
            sseNotSplit = np.sum(dataSet[clusterAssment[:, 0].A1 != i])
            # 该簇分裂之后的 SSE
            sseSplit = np.sum(splitClustAss[:, 1])
            # 总的 SSE
            totalSSE = sseNotSplit + sseSplit
            # 若满足条件则更新一系列操作，最终遍历所有簇之后，得到最佳的分裂簇
            if totalSSE < lowestSSE:
                bestCentToSplit = i
                bestNewCents = newCentroids
                bestClustAss = splitClustAss.copy()
                lowestSSE = totalSSE
        # 更新待分裂簇分裂后的簇标签（默认都是0,1，要更改成全局的标签）
        # 0 -> i
        # 1 -> new(centroids.shape[0])
        bestClustAss[bestClustAss[:, 0].A1 == 1, 0] = centroids.shape[0]
        bestClustAss[bestClustAss[:, 0].A1 == 0, 0] = bestCentToSplit
        # 将处理好的待分裂簇数据更新到全体数据上
        clusterAssment[clusterAssment[:, 0].A1 == bestCentToSplit, :] = bestClustAss
        # 更新并添加簇质心矩阵
        centroids[bestCentToSplit, :] = bestNewCents[0, :]
        centroids = np.row_stack((centroids, bestNewCents[1, :]))
    return centroids, clusterAssment

dataSet2 = loadDataSet(path+"testSet2.txt")
biKmeans(dataSet2, k=3)

