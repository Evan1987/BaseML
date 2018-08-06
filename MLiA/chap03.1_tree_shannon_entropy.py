
# coding: utf-8
import numpy as np
import pandas as pd
import math

# 计算香农熵
def calcShannonEnt(dataSet):
    totalNum = len(dataSet)
    labels = pd.Series([x[-1] for x in dataSet])
    labelsSummary = labels.groupby(labels).size().apply(lambda x :(-x/totalNum)*math.log2(x/totalNum))
    result =labelsSummary.sum()
    return result

