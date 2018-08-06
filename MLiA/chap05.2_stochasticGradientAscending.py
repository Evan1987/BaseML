
# coding: utf-8


import pandas as pd
import numpy as np

path = "F:/for learn/MLiA/Ch05/"

def loadDataSet(path,outtype="mat"):
    df = pd.read_table(path+"testSet.txt",header=None)
    df.columns = ["x1","x2","label"]
    df["x0"] = 1
    if(outtype=="df"):
        return df
    else:        
        dataMat = np.mat(df.loc[:,["x0","x1","x2"]])
        # since df.label is only one pd.Series np.mat just generate a row vec
        labelMat = np.mat(df.label).transpose()
        return dataMat,labelMat

def sigmoid(x):
    return 1/(1+np.exp(-x))

# gradient ascending opt in Logistic
def stocGradAscend(dataMat,labelMat,initW = np.ones((dataMat.shape[1],1)),alpha=0.01):
    # create initial weight vec as a tall column vec
    m,n = dataMat.shape
    w = initW
    for i in range(m):
        yHat = sigmoid(dataMat[i]*w)
        error = labelMat[i]-yHat
        w = w + alpha * dataMat[i].transpose()*error
    return w   

# improved gradient ascending
def impStocGradAscend(dataMat,
                      labelMat,
                      initW = np.ones((dataMat.shape[1],1)),
                      numIter=150):
    import random
    import math
    m,n = dataMat.shape
    w = initW
    for i in range(numIter):
        dataIndex = np.arange(m).tolist()
        for j in range(m):
            # alpha descents with loops going on
            alpha = 4/(1+i+j)+0.01
            # pick record randomly each time
            randIndex = math.floor(random.uniform(0,len(dataIndex)-0.001))
            yHat = sigmoid(dataMat[randIndex]*w)
            error = labelMat[randIndex]-yHat
            w = w + alpha * dataMat[randIndex].transpose()*error
            del dataIndex[randIndex]
    return w


# In[16]:

def calPrecision(dataMat,df,w):
    yHat = sigmoid(dataMat*w)
    df["pred"] = yHat
    df["predLabel"] = (df.pred>=0.5).astype(int)
    return len(df[df.label==df.predLabel])/len(df)



dataMat,labelMat = loadDataSet(path)
df = loadDataSet(path,outtype="df")


w = stocGradAscend(dataMat,labelMat)
resultList=[]

# 对全部记录运行200次，记录分类效果的变化
for i in range(200):
    w = stocGradAscend(dataMat,labelMat,initW=w)
    resultList.append(calPrecision(dataMat,df,w))
resultList


w = impStocGradAscend(dataMat,labelMat)
calPrecision(dataMat,df,w)

