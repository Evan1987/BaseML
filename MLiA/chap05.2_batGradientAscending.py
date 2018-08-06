
# coding: utf-8

# In[181]:

import numpy as np
import pandas as pd

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
def gradAscent(dataMat,labelMat,alpha=0.001,maxcycles=500):
    # create initial weight vec as a tall column vec
    w = np.ones((dataMat.shape[1],1))
    yHat = sigmoid(dataMat*w)
    error = labelMat-yHat
    stepVec = alpha * dataMat.transpose()*error
    #step = np.max(abs(stepVec))
    i = 1
    while i<=maxcycles:
        w = w + stepVec
        yHat = sigmoid(dataMat*w)
        error = labelMat-yHat
        stepVec = alpha * dataMat.transpose()*error
        #step = np.max(abs(stepVec))
        i += 1
    print("opt cycles num is %d"%i)
    return w   


# In[237]:

dataMat,labelMat = loadDataSet(path)

df = loadDataSet(path,outtype="df")

w = gradAscend(dataMat,labelMat)

yHat = sigmoid(dataMat*w)

df["pred"] = yHat

df["predLabel"] = (df.pred>=0.5).astype(int)

len(df[df.label==df.predLabel])/len(df)

