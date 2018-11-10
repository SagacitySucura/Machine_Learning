import numpy as np
from itertools import product
from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt 

iris = datasets.load_iris()
X = iris['data'][:, [0, 2]]
y = iris['target']
#决策树有很多参数，一般我们默认就行了，当参数很多时最好自己先降维而不是用函数
#自带的max_features参数
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#np.meshgrid函数生成的结果，前者是整体的复制，后者是维度内部的复制，为的就是打散后能够
#组成网格点，比如x=y=[1,2],则经过函数meshgrid之后，x=[[1,2],[1,2]],整体复制一遍而已，
#y=[[1,1],[2,2]],显而易见，每个维度复制一遍而已（具体复制的次数是根据x与y的维度而定的）
#经过ravel函数打散后 x=[1 2 1 2],y=[1 1 2 2],很容易可以看出来这样就可以完成x与y的组合了
#np.c_[]就是这样的组合函数
xv, yv = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xv.ravel(), yv.ravel()])
Z = Z.reshape(xv.shape)
#这个函数用于画等高线，这里高度代表花的可能性，类别一样的花处于相同的高度，所以颜色相同
#Z是把所有的可能性都描述出来
plt.contourf(xv, yv, Z, alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=1)
plt.show()