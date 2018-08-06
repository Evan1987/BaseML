
# coding: utf-8
import numpy as np

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

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
    return list(map(frozenset, C1))

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
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# Lk: 项长度为 k 的频繁项集, [frozenset]
# 生成 项长度为 k+1 的候选项集： Cnew
def aprioriGen(Lk):
    # 最后生成候选项的容器
    retList = []
    # 当前频繁项的项长度
    k = len(Lk[0])
    # 生成的候选集的项长度： 在当前频繁项长度基础上 +1
    genLen = k + 1
    # 当前频繁项集的元素数量
    lenLk = len(Lk)
    # 遍历所有频繁项交叉组合
    # 若要生成长度为 k+1的候选项，则只需组合频繁项前 k-1个元素相同的两项即可
    # 如若要在 L2：｛0,1｝，｛0,2｝，｛1,2｝，生成 C3，只需组合｛0,1｝和｛0,2｝即可
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            # 一定要对齐比较
            Lk_i_sub = list(Lk[i])[:(k-1)]
            Lk_j_sub = list(Lk[j])[:(k-1)]
            Lk_i_sub.sort()
            Lk_j_sub.sort()
            if Lk_i_sub==Lk_j_sub:
                retList.append(frozenset(Lk[i]|Lk[j]))
    return retList



def apriori(dataSet, minSupport=0.5):
    # 长度为1的候选集
    C1 = createC1(dataSet)
    # 实际项集，用来算支持度
    D = list(map(set, dataSet))
    # 生成 L1频繁项集 和支持度数据    
    L1, supportData = scanD(D, C1, minSupport=minSupport)
    # 一个包含不同阶频繁项的集合：[L1,L2,...Lk...]
    L = []
    # 表示最新的频繁项集
    Lk = L1
    L.append(Lk)
    # 当最新的频繁项集为空时，循环停止
    while(len(Lk)>1):
        Ck = aprioriGen(Lk)
        Lk, supk = scanD(D, Ck, minSupport=minSupport)
        # 记录各阶项集的支持度
        supportData.update(supk)
        L.append(Lk)
    return L, supportData


# 计算规则的可信度，并输出有效的候选后件 y 集，过程中不断累加 bigRulesList
# 对于n阶频繁项生成的规则是 n-1 -> 1
# freqSet：某阶频繁项集 L[i] 的元素，频繁项，set
# H：后件 y 集，每个元素都是freqSet的真子集，元素长度都为 1
# supportData：记录所有项支持度的dict
# bigRuleList：包含关联规则的元组列表 List[(x, y, conf)]
# minConf：最小可信度阈值
def calConf(freqSet, H, supportData, bigRuleList, minConf=0.7):
    # 记录后件集 H中最终满足有效关联规则的 后件
    prunedH = []
    # y：后件 set
    for y in H:
        # freqSet-y : 前件 x set
        x = freqSet - y
        # conf(x->y) = supp(x | y) / supp(x)
        conf = supportData[freqSet]/supportData[x]
        if conf >= minConf:
            print(x, "-->", y, "conf:", conf)
            # 经过此函数，bigRuleList会改变
            bigRuleList.append((x, y, conf))
            prunedH.append(y)
    return prunedH

# 递归地产生更多的关联规则
# freqSet：某阶频繁项集 L[i] 的元素，set
# H：后件 y 集，每个元素都是freqSet的真子集，元素长度都为 1
# supportData：记录所有项支持度的dict
# bigRuleList：包含关联规则的元组列表 List[(x, y, conf)]
# minConf：最小可信度阈值
def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf=0.7):
    # 后件集每项的长度
    m = len(H[0])
    # 如果当前频繁项元素的数量比后件集的元素长度（初始 H 中的元素长度都为 1）大 2以上
    # 例如 freqSet = ｛A，B，C｝长度为3，而 H = [A,B,C] 元素长度为 1
    # 则 freqSet完全可以支撑 项｛A，B｝的关联规则挖掘
    if(len(freqSet) > (m+1)):
        # 生成长度为 len(H[0])+1 (m+1)的候选项集
        Hmp1 = aprioriGen(H)
        # 产生满足条件的有效后件
        Hmp1 = calConf(freqSet, Hmp1, supportData, bigRuleList, minConf)
        # 如果 Hmp1还能继续组合，则递归进入下一个
        if(len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)


# 关联规则生成函数，包含关联规则(x--conf-->y)的元组列表 List[(x, y, conf)]
# L：频繁项集
# supportData:各项集的支持度信息 dict
# minConf ： 最小置信度阈值
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # 从2阶频繁项集遍历，因为 1阶不存在关联
    for i in range(1, len(L)):
        # freqSet 为该阶频繁项集的一个元素，frozenSet[item]
        for freqSet in L[i]:
            # 将freqSet的每一个item形成一个独立的freqSet，并以列表形式容纳
            # H1中的每个元素都是该freqSet的真子集
            H1 = [frozenset([item]) for item in freqSet]
            # 如果频繁项集是 2阶的，那么只能生成 1->1的关联规则
            if(i == 1):
                calConf(freqSet, H1, supportData, bigRuleList, minConf)
            # 如果频繁项集是更高阶的，那么就可以生成多层规则
            # 如 3阶的可生成 1->2 或 2->1，所以调用方式不同
            else:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


dataSet = loadDataSet()
L, supportData = apriori(dataSet, minSupport=0.5)
rules = generateRules(L, supportData, minConf=0.6)

