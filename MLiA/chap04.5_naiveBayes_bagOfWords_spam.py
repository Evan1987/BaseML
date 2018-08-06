
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd
import string
import math
import re
import os
import random


# In[8]:

path = "F:/for learn/MLiA/Ch04/email/"


# In[13]:

def readFilesBat(path):
    fileNames = os.listdir(path+"spam")
    spamNum = len(fileNames)
    spamLogs = [open(path+"spam/"+x,errors="ignore").read() for x in fileNames]
    
    fileNames = os.listdir(path+"ham")
    hamNum = len(fileNames)
    hamLogs = [open(path+"ham/"+x,errors="ignore").read() for x in fileNames]
    
    labels = np.r_[[1]*spamNum,[0]*hamNum].tolist()
    totalLogs = spamLogs + hamLogs
    return totalLogs,labels
 # 邮件切分文本
def textParse(bigString):
    listOfTokens = re.split("\\W*",bigString)
    result = [tok.lower() for tok in listOfTokens if len(tok)>2]
    return result
# 划分训练集和测试集
def logSplit(totalSet,labels,trainRatio):
    num = len(labels)
    sampleIndex = random.sample(range(num),int(trainRatio*num))
    totalSet = pd.Series(totalSet)
    labels = pd.Series(labels)
    trainSet = totalSet[sampleIndex].tolist()
    trainLabels = labels[sampleIndex].tolist()
    testSet = totalSet[totalSet.index.isin(sampleIndex)==False].tolist()
    testLabels = labels[labels.index.isin(sampleIndex)==False].tolist()
    return [(trainSet,trainLabels),(testSet,testLabels)]


# In[47]:

class spamTest():
    def __init__(self,totalLogs,labels,trainRatio):
        self.totalLogs = totalLogs
        self.labels = labels
        self.trainRatio = trainRatio
        self.totalSet = [textParse(x) for x in totalLogs]
        [(self.trainSet,self.trainLabels),(self.testSet,self.testLabels)] =         logSplit(self.totalSet,labels,trainRatio)
        self.vocabList = self.createVocabList(trainSet=self.trainSet)
    # 创建语料库
    def createVocabList(self,trainSet):
        x = np.hstack(trainSet).tolist()
        return list(set(x))
    # 文档词袋
    def bagOfWords2Vec(self,vocabList,inputSet):
        x = pd.Series(inputSet)
        xSummary = x.groupby(x).size().reindex(vocabList).replace(np.nan,0).astype("int").tolist()
        return xSummary
    # 文档词集
    def setOfWords2Vec(self,vocabList,inputSet):
        return np.isin(vocabList,inputSet).astype(int).tolist()
    # 训练函数
    def trainNB0(self,trainSet,trainLabels,vocabList):
        pClass1 = sum(trainLabels)/len(trainLabels)
        trainMat = [self.bagOfWords2Vec(vocabList,x) for x in trainSet]
        def wordsConditionProb(trainMat,trainLabels,label):
            trainLabels = pd.Series(trainLabels)
            dataSummary = np.array(pd.Series(trainMat)[list(trainLabels[trainLabels==label].index)].tolist())
            wordsNumStat = np.apply_along_axis(arr=dataSummary,axis=0,func1d=sum)
            wordsNumTotal = sum(np.apply_along_axis(arr=dataSummary,axis=1,func1d=sum))
            # Laplace Smoothing: the num of classes in this case is 2
            return (wordsNumStat+1)/(wordsNumTotal+2)
        [p0Vec,p1Vec] = [np.log(wordsConditionProb(trainMat,trainLabels,x)) for x in [0,1]]
        return p0Vec,p1Vec,pClass1
    # 预测函数
    def classifyNB(self,vec2Classify,p0Vec,p1Vec,pClass1):
        p0 = sum(vec2Classify*p0Vec)+math.log(1-pClass1)
        p1 = sum(vec2Classify*p1Vec)+math.log(pClass1)
        result = 0 if p0>=p1 else 1
        return result
    # 封装函数，直接根据训练集和待预测记录进行预测
    def neatClassifyNB(self,vec,trainSet,trainLabels):
        vocabList = self.createVocabList(trainSet)
        p0Vec,p1Vec,pClass1 = self.trainNB0(trainSet,trainLabels,vocabList)
        vec2Classify = self.setOfWords2Vec(vocabList,vec)
        result = self.classifyNB(vec2Classify,p0Vec,p1Vec,pClass1)
        return result


# In[48]:

totalLogs,labels = readFilesBat(path)


# In[49]:

x = spamTest(totalLogs,labels,0.8)


# In[50]:

forecastLabels = [x.neatClassifyNB(vec=z,
                                   trainSet=x.trainSet,
                                   trainLabels=x.trainLabels) for z in x.testSet]
forecastLabels


# In[51]:

x.testLabels

