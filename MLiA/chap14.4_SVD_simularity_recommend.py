
# coding: utf-8

import numpy as np
import numpy.linalg as la

def loadExData():
    x = [[4, 4, 0, 2, 2],
         [4, 0, 0, 3, 3],
         [4, 0, 0, 1, 1],
         [1, 1, 1, 2, 0],
         [2, 2, 2, 0, 2],
         [1, 1, 1, 0, 0],
         [5, 5, 5, 0, 0]]
    return np.mat(x)
def loadExData2():
    x =[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
        [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
        [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
        [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    return np.mat(x)

def eculidSim(inA, inB):
    sim = 1/(1 + la.norm(inA - inB))
    return sim
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]
def cosSim(inA, inB):
    prod = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (prod / denom)

# 基于物品相似度的推荐
# user:目标用户
# item:目标物品
def standEst(datMat, user, simMeas, item):
    # 物品数量
    n = np.shape(datMat)[1]
    # 遍历所有该用户评过分的物品，计算目标物品与该物品的相似度
    # 目标物品的打分（加权平均） = sum(相似度i * 该用户打分i)/sum(相似度i)
    # 计算分母的累积值
    simTotal = 0
    # 计算分子的累积值
    ratSimTotal = 0
    for j in range(n):
        userRating = datMat[user, j]
        if userRating == 0:
            continue
        # 筛出目标物品与其他商品打分重叠（被同一用户打分）的记录
        overLap = np.argwhere(datMat[:, item].A1 * datMat[:, j].A1 > 0).flatten()
        if len(overLap) == 0:
            similarity = 0
        else:
            # 如果有重叠的，则计算目标物品与当前物品的相似度
            similarity = simMeas(datMat[overLap, item], datMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity*userRating 
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

# 对用户未打分的物品进行评价打分，然后从中推荐出打分最高的 N 个物品
def recommend(datMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.argwhere(datMat[user, :].A1 == 0).flatten()
    if len(unratedItems) == 0: return "%s rated everythig"%user
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(datMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    sortedItems = sorted(itemScores, key=lambda z: z[1], reverse=True)
    return sortedItems[:N]


def svdEst(datMat, user, simMeas, item, threhold=0.9):
    n = np.shape(datMat)[1]
    # 遍历所有该用户评过分的物品，计算目标物品与该物品的相似度
    # 目标物品的打分（加权平均） = sum(相似度i * 该用户打分i)/sum(相似度i)
    # 计算分母的累积值
    simTotal = 0
    # 计算分子的累积值
    ratSimTotal = 0
    
    # 根据阈值选择降维所至的维数
    U, Sigma, Vt = la.svd(datMat)
    Sigma2 = np.power(Sigma, 2)
    totalCon = sum(Sigma2)
    cumConRatio = np.cumsum(Sigma2)/totalCon
    selectNum = np.argwhere(cumConRatio > threhold).flatten()[0] + 1
    # 根据Vt生成描述物品相似度的降维矩阵，后续评判物品相似度都是在这个矩阵中进行
    # 每行为物品，每列为隐藏因子
    itemsMat = Vt[:selectNum, :].T
    for j in range(n):
        userRating = datMat[user, j]
        if userRating == 0 or j == item: continue
        # 由于输入计算相似度的是列向量，所以这里行向量需要加转置
        similarity = simMeas(itemsMat[item, :].T, itemsMat[j, :].T)
        simTotal += similarity
        ratSimTotal += simTotal * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

datMat = loadExData2()
recommend(datMat, 1, estMethod=svdEst)

