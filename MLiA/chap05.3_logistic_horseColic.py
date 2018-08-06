
# coding: utf-8

# In[43]:

import numpy as np
import pandas as pd

def loadDataSet(filePath):
    df = pd.read_table(filePath,header=None)
    labelMat = np.mat(df.loc[:,df.columns[-1]]).transpose()
    data = df.loc[:,df.columns[:-1]]
    data["x0"] = 1
    dataMat = np.mat(data)
    return dataMat,labelMat
# improved sigmoid to avoid the overflow case of big x
def sigmoid(x):
    sign = 1 if x>0 else -1
    x = sign*50 if abs(x)>50 else x
    return 1/(1+np.exp(-x))
def classifyVector(x,w):
    prob = sigmoid(x*w)
    result = 1 if prob>0.5 else 0
    return result
# improved gradient ascending
def impStocGradAscend(dataMat,
                      labelMat,
                      numIter=150):
    import random
    import math
    m,n = dataMat.shape
    w = np.ones((dataMat.shape[1],1))
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
# calc pred precision
def calPrecision(testDataMat,testLabelMat,w):
    predLabel = np.array([classifyVector(x,w) for x in testDataMat])
    testLabel = np.array(testLabelMat).ravel().tolist()
    precision = len(np.where(predLabel==testLabel)[0])/len(predLabel)
    return precision


# In[44]:

path = "F:/for learn/MLiA/Ch05/"
trainDataMat,trainLabelMat = loadDataSet(path+"horseColicTraining.txt")
testDataMat,testLabelMat = loadDataSet(path+"horseColicTest.txt")


# In[48]:

w = impStocGradAscend(trainDataMat,trainLabelMat)


# In[50]:

calPrecision(testDataMat,testLabelMat,w)

