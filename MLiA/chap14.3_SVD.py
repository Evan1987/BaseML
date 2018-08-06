
# coding: utf-8
import numpy as np
def loadData():
    x = [[1, 1, 1, 0, 0],
         [2, 2, 2, 0, 0],
         [1, 1, 1, 0, 0],
         [5, 5, 5, 0, 0],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 3, 3],
         [0, 0, 0, 1, 1]]
    return np.mat(x)

def slimDatBySVD(dat, threhold=0.9):
    U,Sigma,Vt = np.linalg.svd(dat)
    index = np.argwhere(np.cumsum(Sigma)/np.sum(Sigma) >= threhold).flatten()[0]+1
    newSigma = np.mat(np.diag(Sigma[:index]))
    newDat = U[:, :index] * newSigma * Vt[:index, :]
    return np.mat(np.around(newDat, decimals=2))

dat = loadData()
newDat = slimDatBySVD(dat)
newDat




