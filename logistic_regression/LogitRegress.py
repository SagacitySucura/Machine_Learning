import numpy as np


def loadDataSet():
    dataMat = []
    dataLabel = []
    f = open('logistic_regression/testSet.txt')
    for line in f.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #这个1.0是偏置
        dataLabel.append(int(lineArr[2]))
    return dataMat, dataLabel


def sigmod(x):
    return 1.0/(1+np.exp(-x))


#可以看见梯度下降法每次都需要拿整个矩阵计算一次，当数据矩阵非常庞大时就会
#非常影响速度
def gradDesc(dataIn, labelIn, maxCycles):                  #虽然形式上类似梯度上升，其实是根据西瓜书3.27
    dataMat = np.mat(dataIn)                    #的式子得到的结果，他将式子的求极大值变成了
    labelMat = np.mat(labelIn).transpose()      #求相反数的极小值，所以可以采用梯度下降
    m, n = np.shape(dataMat)
    alpha = 0.001                               # 学习率                            # 最大迭代次数
    weights = np.ones((n, 1))                   # 权重w
    for k in range(maxCycles):
        h = sigmod(dataMat*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return np.array(weights)


#这里采用的随机梯度下降，可以明显的看出来我们并没有直接采用所有的样本，而是随机的
#选取一些样本参与计算梯度,切记py里面的array的乘法得到的结果仍然是array而并不是我们
#想要的内积，所以要求内积需要用上np.sum()函数来求和array里面的所有数。
def stocGradDesc(dataIn, labelIn, iterNum):
    dataArray = np.array(dataIn)
    labelArray = np.array(labelIn)
    m, n = np.shape(dataArray)
    weights = np.ones(n)
    for i in range(iterNum):
        for j in range(m):
            alpha = 4.0/(i + j + 1.0) + 0.0001
            randIndex = np.random.randint(0, m)
            h = sigmod(np.sum(weights * dataArray[randIndex]))
            error = labelArray[randIndex] - h
            weights = weights + alpha * error * dataArray[randIndex]
    return weights


def newTon(dataIn, labelIn, iterNum):
    dataArray = np.array(dataIn)
    labelArray = np.array(labelIn)
    m, n = np.shape(dataArray)
    weights = np.ones((n))
    for i in range(iterNum):
        for j in range(m):
            h = sigmod(np.sum(weights * dataArray[j]))
            up = (labelArray[j] - h) * dataArray[j]
            down = np.sum(dataArray[j] * \
                   dataArray[j] * h * (1 - h))
            weights = weights + up
    return np.array(weights)


def plotResult(weights):
    import matplotlib.pyplot as plt 
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] 
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.linspace(-4, 4)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


#由结果可以显而易见，随机梯度下降的迭代次数明显要小于常规的梯度下降    
def main():
    data, label = loadDataSet()
    weights = newTon(data, label, 500)
    plotResult(weights)


if __name__ == '__main__':
    main()