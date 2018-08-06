
# coding: utf-8

import numpy as np
import pandas as pd

path = "F:/for learn/MLiA/Ch06/"
# data loading func
def loadDataSet(filePath):
    df = pd.read_table(filePath, header=None)
    labelMat = np.mat(df.loc[:, df.columns[-1]]).transpose()
    data = df.loc[:, df.columns[:-1]]
    dataMat = np.mat(data)
    return dataMat, labelMat
# select pair of alpha's index randomly(not heuristicly)
def selectJrand(i, m):
    import random
    j=i
    while(j==i):
        j = int(random.uniform(0, m))
    return j
# valid alpha value
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    else:
        pass
    return aj        

# toler>0 and small to avoid float trap
# this func is based on the Platt's equation: ui = wTxi-b
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    ### Preparation
    # type transformation
    dataMatrix = np.mat(dataMatIn)
    m, n =dataMatrix.shape
    # transform to a column vec
    labelMat = np.mat(classLabels).reshape(len(classLabels), -1)
    # original para setting
    b = 0
    alphas = np.mat(np.zeros((m, 1)))
    iterIndex = 0
    # iteration start
    while(iterIndex<maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # fXi==>ui = wTxi-b
            fXi = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T) - b)
            # Ei = ui - yi
            # KKT Complementary condition: 
            # 1. yiui>=1,if alphai=0 => yiui-1>=0 => yi(ui-yi)>=0 => yiEi>=0
            # 2. yiui<=1,if alphai=C => yiEi<=0
            # 3. yiui==1,if alphai belong (0,C) => yiEi=0
            # other condition: alphai belong [0,C] 
            Ei = fXi - float(labelMat[i])
            # the alphai which violates the KKT condition needs to be updated
            if ((labelMat[i]*Ei > toler) and alphas[i] > 0) or ((labelMat[i]*Ei < -toler) and alphas[i] < C):
                # select another alpha to form a pair
                j = selectJrand(i, m)
                # yi and yj have different sign
                if(labelMat[i]!=labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                # yi and yj have same sign
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                # L==H means the feasible domain(a line in the [0,C] rectangle) is only a point
                # which means alphai and alphaj are both already on the boundary 
                # no need for further updating
                if(L==H):
                    print("L==H")
                    continue
                fXj = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T) - b)
                Ej = fXj - float(labelMat[j])
                alphaIOld = alphas[i].copy()
                alphaJOld = alphas[j].copy()

                # eta = kii+kjj-2kij
                eta = float(dataMatrix[i, :] * dataMatrix[i, :].T +
                            dataMatrix[j, :] * dataMatrix[j, :].T -
                            2*dataMatrix[i, :]*dataMatrix[j, :].T)
                # eta<0 means target func has no minimals
                # eta=0 means target func is a linear func
                if eta <= 0:
                    print("eta<=0")
                    continue
                alphas[j] += labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(aj=alphas[j], H=H, L=L)
                deltaAlphaJ = alphaJOld - alphas[j]
                if(abs(deltaAlphaJ)<0.00001):
                    print("j not moving enough")
                    continue
                # ai = ai + yiyj(aj-ajnew)
                alphas[i] += labelMat[i] * labelMat[j] * deltaAlphaJ
                # update b
                b1 = Ei + labelMat[i]*(alphas[i]-alphaIOld)*(dataMatrix[i, :]*dataMatrix[i, :].T)+\
                     labelMat[j]*(alphas[j]-alphaJOld)*(dataMatrix[i, :]*dataMatrix[j, :].T)+b
                b2 = Ej + labelMat[i]*(alphas[i]-alphaIOld)*(dataMatrix[i, :]*dataMatrix[j, :].T)+\
                     labelMat[j]*(alphas[j]-alphaJOld)*(dataMatrix[j, :]*dataMatrix[j, :].T)+b
                if(alphas[i]>0 and alphas[i]<C):
                    b = b1
                elif(alphas[j]>0 and alphas[j]<C):
                    b = b2
                else:
                    b = 0.5*(b1+b2)
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" %(iterIndex, i, alphaPairsChanged))
        # no alpha change in this iter
        # if iter nums>=maxIter means convergence
        if alphaPairsChanged == 0:
            iterIndex += 1
        else:
            iterIndex = 0
        print("iteration number: %d" %iterIndex)
    return b, alphas

dataMat, labelMat = loadDataSet(path+"testSet.txt")
b, alphas = smoSimple(dataMatIn=dataMat, classLabels=labelMat, C=0.6, toler=0.001, maxIter=40)

float(b)
# support vector
alphas[alphas > 0]

