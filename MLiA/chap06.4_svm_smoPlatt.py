
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

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = np.mat(dataMatIn)
        self.labelMat = np.mat(classLabels).reshape(len(classLabels), -1)
        self.C = C
        self.toler = toler
        self.m = self.X.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # cache matrix of errors: m*2
        # column0: isValid(optimized): 0|1
        # column1: Ei
        self.eCache = np.mat(np.zeros((self.m, 2)))
    
    # cal the Ek
    # fXk = uk = wTxk-b
    def calcEk(self, k):
        fXk = float(np.multiply(self.labelMat, self.alphas).T * self.X * self.X[k, :].T - self.b)
        Ek = fXk - float(self.labelMat[k])
        return Ek
    
    # valid alpha value
    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        elif aj < L:
            aj = L
        else:
            pass
        return aj 
    
    # select pair of alpha's index heuristicly by Platt's method
    # in which j makes the greatest |Ei-Ej|
    def selectJ(self, i, Ei):
        self.eCache[i] = [1, Ei]
        validTmp = np.nonzero(self.eCache[:, 0].A)[0]   # np.mat.A returns an array
        validEcacheList = validTmp[validTmp != i]
        if(len(validEcacheList) > 0):
            EkList = np.array([self.calcEk(k) for k in validEcacheList])
            deltaEList = np.abs(EkList-Ei)
            maxK = deltaEList.argmax()
            j = validEcacheList[maxK]
            Ej = EkList[maxK]            
        else:
            # select pair of alpha's index randomly(not heuristicly)
            def selectJrand(i, m):
                import random
                j = i
                while(j==i):
                    j = int(random.uniform(0, m))
                return j
            j = selectJrand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej
    
    # func to update the k row of eCache matrix
    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]
        
    # the Inner loop of Platt's SMO
    def innerL(self, i):
        Ei = self.calcEk(i)
        if (self.labelMat[i]*Ei < -self.toler and self.alphas[i] < self.C) or \
                (self.labelMat[i]*Ei > self.toler and self.alphas[i] > 0):
            # select another alpha to form a pair
            j, Ej = self.selectJ(i, Ei)
            # yi and yj have different sign
            if(self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            # yi and yj have same sign
            else:
                L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                H = min(self.C, self.alphas[i] + self.alphas[j])
            # L==H means the feasible domain(a line in the [0,C] rectangle) is only a point
            # which means alphai and alphaj are both already on the boundary 
            # no need for further updating
            if(L==H):
                print("L==H")
                return 0
            
            alphaIOld = self.alphas[i].copy()
            alphaJOld = self.alphas[j].copy()

            # eta = kii+kjj-2kij
            eta = float(self.X[i, :] * self.X[i, :].T +
                        self.X[j, :] * self.X[j, :].T -
                        2*self.X[i, :] * self.X[j, :].T)
            # eta<0 means target func has no minimals
            # eta=0 means target func is a linear func
            if eta <= 0:
                print("eta<=0")
                return 0
            self.alphas[j] += self.labelMat[j]*(Ei-Ej)/eta
            self.alphas[j] = self.clipAlpha(aj=self.alphas[j], H=H, L=L)
            self.updateEk(j)
            deltaAlphaJ = alphaJOld - self.alphas[j]
            if(abs(deltaAlphaJ)<0.00001):
                print("j not moving enough")
                return 0
            # ai = ai + yiyj(aj-ajnew)
            self.alphas[i] += self.labelMat[i] * self.labelMat[j] * deltaAlphaJ
            self.updateEk(i)
            # update b
            b1 = Ei + self.labelMat[i]*(self.alphas[i]-alphaIOld)*(self.X[i, :]*self.X[i, :].T) +\
                 self.labelMat[j]*(self.alphas[j]-alphaJOld)*(self.X[i, :]*self.X[j, :].T) + self.b
            b2 = Ej + self.labelMat[i]*(self.alphas[i]-alphaIOld)*(self.X[i, :]*self.X[j, :].T) +\
                 self.labelMat[j]*(self.alphas[j]-alphaJOld)*(self.X[j, :]*self.X[j, :].T) + self.b
            if(self.alphas[i]>0 and self.alphas[i]<self.C):
                self.b = b1
            elif(self.alphas[j]>0 and self.alphas[j]<self.C):
                self.b = b2
            else:
                self.b = 0.5*(b1+b2)
            return 1
        else:
            return 0

# the outer loop of Platt's SMO
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ('lin',0)):
    # create object initially
    oS = optStruct(dataMatIn=dataMatIn,
                   classLabels=classLabels,
                   C=C,
                   toler=toler)
    # set params initially
    iterIndex = 0
    alphaPairsChanged = 0
    entireSet = True
    # iteration never stops unitl reach maxIter as long as it is still updating
    while (iterIndex<maxIter) and (alphaPairsChanged>0 or entireSet):
        alphaPairsChanged = 0
        # for the first scan usually
        # update each alpha which violates the KKT condition
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i)
            print("fullSet, iter: %d i: %d, pairs changed %d" % (iterIndex, i, alphaPairsChanged))
        # the updated alpha which is on the boundary will not be updated any more
        # so just focus on the alphas in the boundary.
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i)
                print("nonBound, iter: %d i: %d, pairs changed %d" % (iterIndex, i, alphaPairsChanged))
        iterIndex += 1
        # if the iteration scaned the totalSet just now
        # the next iteration will not do that again unless no alpha was updated
        if entireSet:
            entireSet = False
        # if no alpha was updated, then scan the totalSet once again.
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iterIndex)
    return oS.b, oS.alphas

# calc w
def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr)
    m, n = X.shape
    labelMat = np.mat(classLabels).reshape(m, -1)
    # calc the list(alphai*yi) m*1
    alphaY = np.multiply(alphas, labelMat)
    # calc the list(alphai*yi*xi) m*n
    alphaYX = np.multiply(alphaY, X)
    # calc the sum of list(alphai*yi*xi) np.apply returns an array
    w = np.apply_along_axis(arr=alphaYX, axis=0, func1d=np.sum)
    # change into 1*n (row vec)
    return np.mat(w).reshape(1, n)

# classify func for input dataArr
def classify(w,b,dataArr):
    X = np.mat(dataArr)
    m, n = X.shape
    w = np.mat(w).reshape(1, n)
    # main calc part
    resultTmp = X*w.T-b
    # main classify part
    f = lambda x: 1 if x>=0 else -1
    # return a array with length=X.shape[0]
    result = np.apply_along_axis(arr=resultTmp, axis=1, func1d=f)
    return result

dataMat, labelMat = loadDataSet(path+"testSet.txt")

b, alphas = smoP(dataMat, labelMat, C=0.6, toler=0.001, maxIter=40)
# support vector
print("the support vector num is %d" % len(alphas[alphas > 0].A.ravel()))
w = calcWs(alphas=alphas, dataArr=dataMat, classLabels=labelMat)
trainResult = classify(w, b, dataMat)
actualLabels = labelMat.T.A.ravel()
x = trainResult*actualLabels
# calc train precision
print("the train precision is %f" % float(len(x[x > 0])/len(x)))