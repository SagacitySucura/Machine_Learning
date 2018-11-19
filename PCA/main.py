import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

X = np.array([[-1, -1],
              [-2, -1],
              [-3, -2],
              [1, 1],
              [2, 1],
              [3, 2]])
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签,中文前加上u‘内容’
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.xlabel('x1')
plt.ylabel('y1')
plt.title(u'示例')
plt.scatter(X[:, 0], X[:, 1], c='r', marker='.')
print(type(X))
plt.scatter(pca.transform(X)[:, 0], np.zeros(6), c='g', marker='*')
plt.show()
