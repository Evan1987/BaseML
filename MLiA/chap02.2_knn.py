
# coding: utf-8
import numpy as np
import pandas as pd

def autoNorm(x):
    minX = min(x)
    rangeX = max(x)-min(x)+0.001
    normX = x.apply(func=lambda z: round((z-minX)/rangeX,2))
    return minX,rangeX,normX

def classify0(inX,normdf,k=5):
    normdf["dist"] = normdf[normdf.columns[0:-1]].apply(axis=1,func=lambda x:round(((x-inX)**2).sum()**0.5,2))
    normdf = normdf.sort_values(by="dist",ascending=True)
    classSelect = normdf[:k].groupby("label").count()
    result = normdf[["label"]][:k].groupby("label").size().sort_values(ascending=False).index[0]
    return result

def normX(x,minVars,rangeVars):
    return (x-minVars)/rangeVars

filepath = "F:/for learn/MLiA/Ch02/datingTestSet.txt"
df = pd.read_table(filepath,header=None)
df.columns = ["flyDist","gameTimeRatio","icePerWeek","label"]
normdf = df.copy(deep=True)

summary = normdf[normdf.columns[0:-1]].apply(axis=0,func=autoNorm)

minVars = [round(x[0],2) for x in summary]
rangeVars = [round(x[1],2) for x in summary]
normdf[normdf.columns[0:-1]] = pd.DataFrame(dict(zip(list(summary.index),list(x[2] for x in summary))))

testRatio = 0.3
idx = int(testRatio*df.shape[0])
testRecords = df[:idx]
trainRecords = normdf[idx:]


testRecords["pred"] = testRecords[testRecords.columns[0:3]].apply(axis=1,func=lambda x:classify0(normX(x,minVars,rangeVars),trainRecords,k=5))


testRecords
testRecords.loc[testRecords.pred==testRecords.label]

