
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt


# 1. 基础函数定义
def loadDataSet(fileName, delim="\t"):
    fr = open(fileName)
    datArr = [list(map(float, line.strip().split(delim))) for line in fr.readlines()]
    datMat = np.mat(datArr)
    return datMat

def pca(datMat, topNfeat=-1):
    meanVals = np.mean(datMat, axis=0)
    centralizedDat = datMat - meanVals
    # 生成协方差矩阵，其中指定 列为特征（变量）。默认每一行是一个特征
    # 等价于 centralizedDat.T * centralizedDat/(centralized.shape[0]-1)
    covMat = np.mat(np.cov(centralizedDat, rowvar=False))
    # 求解全部特征值和特征向量，特征值保存在数组中， 特征向量在矩阵中，每一列对应数组中的特征值
    eigVals, eigVecs = np.linalg.eig(covMat)
    # 按升序排列，对应eigVals的次序
    eigValsInd = np.argsort(eigVals)
    if topNfeat>0:
        # 从后向前选取，并逆序
        eigValsInd = eigValsInd[:-(topNfeat+1):-1]
    redEigVecs = eigVecs[:, eigValsInd]
    # 转换到低维空间
    # coordMat：低维坐标矩阵（以主成分向量为基的坐标信息，能够直观体现降维）
    # lowDataMat：转换原坐标矩阵，将坐标基转换为原始坐标基的矩阵，可以与原始矩阵比较变化，无法直观体现降维
    coordMat = centralizedDat * redEigVecs
    lowDataMat = coordMat * (redEigVecs.T) + meanVals
    return coordMat, lowDataMat

path = "F:/for learn/MLiA/Ch13/"
fileName = path + "testSet.txt"
datMat = loadDataSet(fileName)
coordMat, lowDataMat = pca(datMat, topNfeat=1)

# 2. 描绘主成分分析降维后的效果
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datMat[:, 0].flatten().A1, datMat[:, 1].flatten().A1, marker='^', s=90)
ax.scatter(lowDataMat[:, 0].flatten().A1, lowDataMat[:, 1].flatten().A1, marker='o', s=50, c='red')


# 3. 处理半导体数据
def getCleanDat(fileName):
    dat = loadDataSet(fileName, ' ')
    numFeat = dat.shape[1]
    # 用各特征均值替代nan
    for i in range(numFeat):
        x = dat[:, i].A1
        meanVal = np.mean(x[~np.isnan(x)])
        dat[:, i][np.isnan(dat[:, i])] = meanVal
    return dat

fileName = path + "secom.data"
datMat = getCleanDat(fileName)


meanVals = np.mean(datMat, axis=0)
centralizedDat = datMat - meanVals
covMat = np.cov(centralizedDat, rowvar=False)
eigVals, eigVecs = np.linalg.eig(covMat)
eigVals = np.array(list(map(lambda x: x.real, eigVals)))
np.cumsum(eigVals)/np.sum(eigVals)

