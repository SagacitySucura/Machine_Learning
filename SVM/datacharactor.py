import numpy as np 
import matplotlib.pyplot as plt 
import os.path
import re
os.chdir('e:/vscode/machine_learning/SVM/')


def loadData(filename):
    dataMat = []
    labelMat = []
    row = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = re.split("\s+", line.strip())
        for cols in range(len(lineArr) - 1):
            row.append(float(lineArr[cols]))
        dataMat.append(row)
        row = []
        labelMat.append(float(lineArr[-1]))
    return np.array(dataMat), np.array(labelMat)


def main():
    data1, label1 = loadData('traindata.txt')
    positive1 = []
    negative1 = []
    for i in range(len(label1)):
        if(label1[i] == 1):
            positive1.append(data1[i])
        else:
            negative1.append(data1[i])
    positive1 = np.array(positive1)
    negative1 = np.array(negative1)
    data2, label2 = loadData('testdata.txt')
    positive2 = []
    negative2 = []
    for i in range(len(label2)):
        if(label2[i] == 1):
            positive2.append(data2[i])
        else:
            negative2.append(data2[i])
    positive2 = np.array(positive2)
    negative2 = np.array(negative2)
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)  # 创建子图模块
    plt.sca(ax1)    # 设置当前子图为操作模块
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("traindata")
    plt.scatter(positive1[:, 0], positive1[:, 1], c='r', marker='o')
    plt.scatter(negative1[:, 0], negative1[:, 1], c='g', marker='o')
    ax2 = plt.subplot(122)
    plt.sca(ax2)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("test")
    plt.scatter(positive2[:, 0], positive2[:, 1], c='r', marker='o')
    plt.scatter(negative2[:, 0], negative2[:, 1], c='g', marker='o')
    plt.show()


if __name__ == '__main__':
    main()
