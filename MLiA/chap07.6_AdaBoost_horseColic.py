
# coding: utf-8

import numpy as np
import math

def loadSimpData():
    data = [[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]]
    datMat = np.matrix(data)
    classLabels = [1, 1, -1, -1, 1]
    return datMat, classLabels

## 根据阈值进行分类的函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # 输出结果为 m（样本数）*1的向量
    retArray = np.ones((dataMatrix.shape[0], 1))
    if(threshIneq == 'lt'):
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen]>threshVal] = -1
    return retArray
## 单层决策树（树桩stump）生成函数
### 遍历所有stumpClassify函数所有可能的输入值，并找到数据集上最好的单层决策树。
### D：数据样本的权重向量
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    # 对向量来说mat化默认是生成行向量，因此要加 T
    labelMat = np.mat(classLabels).T
    # m：样本数，n：特征数
    m, n = dataMatrix.shape
    numSteps = 10; bestStump = {}; bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # 权重的归一化处理
    D = np.mat(D)
    D = D/D.sum()
    
    for i in range(n):
        # 根据特征i的最小值和最大值计算每次阈值变化的步长
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, numSteps+1):
            for inequal in ['lt', 'gt']:
                # 当前阈值的设置
                threshVal = rangeMin + j*stepSize
                # 得到当前条件下的预测label
                predictVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.vectorize(int)(predictVals != labelMat)
                # 加权错误率，D：数据样本的权重向量
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh inequal: %s,the weighted error is %.3f" % \
                      (i, threshVal, inequal, weightedError))
                
                if(weightedError < minError):
                    minError = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

datMat, classLabels = loadSimpData()
D = np.ones((5, 1))

# AdaBoost过程 train
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m= np.mat(dataArr).shape[0]
    D = np.ones((m, 1))/m
    # 累积汇总结果的容器
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # 在当前情况下输出最好的弱分类器
        # classEst：列向量
        # classLabels：行向量
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        # 分类器的权重值，加入到分类器信息中
        alpha = 0.5 * math.log((1-error)/max(error, 1e-16))
        bestStump['alpha'] = alpha
        # 将这个弱分类器保存起来
        weakClassArr.append(bestStump)
        
        print("classEst: ", classEst.T)
        # 更新样本权重
        # 被正确分类的 D(i) = D(i)*exp(-alpha)，权重被减小 
        # 被错误分类的 D(i) = D(i)*exp(alpha)，权重被增大
        # 最后对所有 D(i)做归一化处理，D(i) = D(i)/sum(D)
        # 以下是利用向量化方式将正确分类和错误分类进行统一的权重计算
        expon = -1*alpha*np.multiply(classEst, np.mat(classLabels).T)# 指数向量，列向量
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        
        # 当前集成后的预测结果（不是最终分类，最终分类是通过结果的符号进行判断的）
        # 模拟最终分类器的输出效果，如果已经可以达到训练集正确分类，那么停止循环
        aggClassEst += alpha*classEst
        print("aggClassEst:", aggClassEst.T)
        
        # 如果错误率已经降到 0 ，则中止
        # 确定分类结果用sign来进行判断
        aggErrors = np.vectorize(int)(np.sign(aggClassEst) != np.mat(classLabels).T)
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate, "\n")
        if errorRate == 0:
            break
    
    return weakClassArr, aggClassEst
        

classifierArray = adaBoostTrainDS(dataArr=datMat, classLabels=classLabels, numIt=30)[0]


# adaBoost 分类函数
# 根据训练结果对任意数据做预测
def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,
                                 classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
    


#--------------------- 在一个大型数据集上应用adaBoost ----------------------------
path = "F:/for learn/MLiA/Ch07/"
trainFile = "horseColicTraining2.txt"
testFile = "horseColicTest2.txt"


# 数据集载入函数
def loadDataSet(fileName):
    file = open(fileName)
    datMat = [];labelMat=[]
    numFeat = len(file.readline().split('\t'))
    for line in file.readlines():
        lineArr = []
        curline = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curline[i]))
        datMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return datMat, labelMat


datArr, labelArr = loadDataSet(path+trainFile)
classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)

testArr, testLabelArr = loadDataSet(path+testFile)
predict = adaClassify(testArr, classifierArr)

error = np.vectorize(int)(np.sign(predict) != np.mat(testLabelArr).T)
errorRate = error.sum()/error.shape[0]
errorRate

#-------------------- 描画ROC曲线并计算AUC -----------------------------

# ROC曲线绘制
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    # 求AUC的辅助量，计算小矩形的面积和
    # 小矩形若沿着X轴移动的话，只移动xStep，所以只要记录每次沿着X轴平移时高度（Y）的累加即可
    ySum = 0
    # 获得升序的元素序号，排名更低（含）的都置为反例，更高的置为正例。
    # 因为是从最小的开始循环，即初始状态都是被置为正例的。
    # FN = TN = 0，TPR = 1，FPR = 1 因此是从右上角开始绘制的。
    sortedIndicies = predStrengths.argsort()
    numPosClass = sum(np.array(classLabels) == 1)
    numNegClass = len(classLabels) - numPosClass
    # 纵坐标为 TPR = TP/(TP+FN),分母为样本中实际正例的数量
    yStep = 1 / numPosClass
    # 横坐标为 FPR = FP/(FP+TN)，分母为样本中实际反例的数量
    xStep = 1 / numNegClass

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 从（1,1）开始绘制，即全部预测值都是正例
    cur = (1, 1)
    for index in sortedIndicies.tolist()[0]:
        # 如果当前作为阈值的实例实际上是正例，则TP数将减少1（因为要被置为反例，导致错误），FN增加1
        # 则相当于 TPR = TP/(TP + FN) 减少了1个单位 -> delY
        if classLabels[index] == 1:
            delX = 0
            delY = yStep
        # 如果当前作为阈值的实例实际上是反例，则FP数将减少1（因为要被置为反例，导致正确），TN增加1
        # 则相当于 FPR = FP/(FP + TN) 减少了1个单位 -> delX
        else:
            delX = xStep
            delY = 0
            # 游标沿着 X轴 平移，故需要记录 Y
            ySum += cur[1]
        curNew = (cur[0] - delX, cur[1] - delY)
        ax.plot([cur[0], curNew[0]], [cur[1], curNew[1]])
        cur = curNew
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for AdaBoost Horse Colic Detection System")
    ax.axis([0, 1, 0, 1])
    auc = ySum * xStep
    print("the Area Under the Curve is: ", auc)
    plt.show()

plotROC(aggClassEst.T, labelArr)

