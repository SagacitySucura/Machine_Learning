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


def gradDesc(dataIn, labelIn):                  #虽然形式上类似梯度上升，其实是根据西瓜书3.27
    dataMat = np.mat(dataIn)                    #的式子得到的结果，他将式子的求极大值变成了
    labelMat = np.mat(labelIn).transpose()      #求相反数的极小值，所以可以采用梯度下降
    m, n = np.shape(dataMat)
    alpha = 0.001                               # 学习率
    maxCycles = 500                             # 最大迭代次数
    weights = np.ones((n, 1))                   # 权重w
    for k in range(maxCycles):
        h = sigmod(dataMat*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights


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
    y = (-weights[0, 0]-weights[1, 0]*x)/weights[2, 0]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    
def main():
    data, label = loadDataSet()
    weights = gradDesc(data, label)
    plotResult(weights)


if __name__ == '__main__':
    main()