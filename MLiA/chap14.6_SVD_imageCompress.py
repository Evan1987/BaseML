
# coding: utf-8

# In[1]:


import numpy as np
import numpy.linalg as la


# In[43]:


def printMat(inMat, thresh=0.8):
    for i in range(32):
        row = ''
        for k in range(32):            
            if float(inMat[i, k]) > thresh:
                row = row + '1'
            else:
                row = row + '0'
        print(row)


# In[41]:


def imgCompress(fileName, numSV=3, thresh=0.8):
    myl = []
    fr = open(fileName)
    for line in fr.readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    datMat = np.mat(myl)
    print("**** original matrix ****")
    printMat(datMat, thresh)
    
    U, Sigma, Vt = la.svd(datMat)
    newSigma = Sigma[:numSV]
    reconMat = U[:,:numSV] * np.mat(np.diag(newSigma)) * Vt[:numSV,:]
    print("**** reconstructed matrix using %d singular values"%numSV)
    printMat(reconMat, thresh)


# In[22]:


path = "F:/for learn/MLiA/Ch14/"
fileName = path + "0_5.txt"


# In[45]:


imgCompress(fileName, numSV=2)

