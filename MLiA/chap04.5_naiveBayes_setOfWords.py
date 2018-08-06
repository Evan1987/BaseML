
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math


# In[100]:

class testNB():
    postingList= [['my','dog','has','flea','problems','help','please'],
                  ['maybe','not','take','him','to','dog','park','stupid'],
                  ['my','dalmation','is','so','cute','I','love','him'],
                  ['stop','posting','stupid','worthless','garbage'],
                  ['mr','licks','ate','my','steak','how','to','stop','him'],
                  ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    
    def __init__(self,trainSet = postingList,trainLabels = classVec):
        self.trainSet = trainSet
        self.trainLabels = trainLabels
        self.vocabList = createVocabList(trainSet)
    # 创建不重复词的列表（语料库）
    def createVocabList(self,trainSet):
        x = np.hstack(trainSet).tolist()
        return list(set(x))    
    # 根据语料库处理每条记录
    def setOfWords2Vec(self,vocabList,inputSet):
        return np.isin(vocabList,inputSet).astype(int).tolist()
    # 计算每个分类ci下wi的条件概率（训练集的输出结果）
    def trainNB0(self,trainSet,classVec,vocabList):
        pAbuse = sum(classVec)/len(classVec)
        trainMat = [setOfWords2Vec(vocabList,x) for x in trainSet]
        def wordsConditionProb(trainMat,classVec,label):
            classVec = pd.Series(classVec)
            dataSummary = np.array(pd.Series(trainMat)[list(classVec[classVec==label].index)].tolist())
            wordsNumStat = np.apply_along_axis(arr=dataSummary,axis=0,func1d=sum)
            wordsNumTotal = sum(np.apply_along_axis(arr=dataSummary,axis=1,func1d=sum))
            # Laplace Smoothing: the num of classes in this case is 2
            return (wordsNumStat+1)/(wordsNumTotal+2)
        [p0Vec,p1Vec] = [np.log(wordsConditionProb(trainMat,classVec,x)) for x in [0,1]]
        return p0Vec,p1Vec,pAbuse
    # 根据训练集的训练结果对目标记录进行预测
    def classifyNB(self,vec2Classify,p0Vec,p1Vec,pClass1):
        p0 = sum(vec2Classify*p0Vec)+math.log(1-pClass1)
        p1 = sum(vec2Classify*p1Vec)+math.log(pClass1)
        result = 0 if p0>=p1 else 1
        return result
    # 封装函数，直接根据训练集和待预测记录进行预测
    def neatClassifyNB(self,vec,trainSet,classVec):
        vocabList = createVocabList(trainSet)
        vec2Classify = setOfWords2Vec(vocabList,vec)
        p0Vec,p1Vec,pClass1 = trainNB0(trainSet,classVec,vocabList)
        return classifyNB(vec2Classify,p0Vec,p1Vec,pClass1)

x = testNB()
x.neatClassifyNB

[classifyNB(x,p0Vec,p1Vec,pClass1=pAbuse) for x in trainMat]

