
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


# In[49]:


# 大小为1的所有候选项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # C1的元素都是长度为1的列表（项）
                C1.append([item])
    C1.sort()
    # 每一项都是不可修改的集合
    return list(map(frozenset,C1))


# In[62]:


# 输出满足最低支持度的频繁项集
# D：实际项集
# Ck：长度为k的候选项集
def scanD(D,Ck,minSupport):
    # 频繁子集的收集容器
    ssCnt = {}
    # 遍历所有实际项集
    for actualSet in D:
        # 遍历所有已有的频繁子集
        for kSet in Ck:
            # 如果当前子集包含于该实际项集中
            if kSet.issubset(actualSet):
                # 若容器里还未收集此子集，则收集，且计数为1
                if not kSet in ssCnt.keys():
                    ssCnt[kSet] = 1
                # 若容器已收集，则计数+1
                else:
                    ssCnt[kSet] += 1
    # 已经全部遍历完毕，需要根据最小支持度阈值来筛选频繁项集
    numItems = len(D)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


# In[65]:


dataSet = loadDataSet()
C1 = createC1(dataSet)
D = list(map(set,dataSet))
L1, suppData0 = scanD(D, C1, 0.5)

