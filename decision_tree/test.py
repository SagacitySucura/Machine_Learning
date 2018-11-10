import numpy as np 
import matplotlib.pyplot as plt 

plt.figure()
x = np.arange(1, 3, 1)
y = np.arange(1, 3, 1)
#meshgrid维度复制
X, Y = np.meshgrid(x, y)
print(X)
print(Y)
#plt.plot(X, Y, color='red', marker='.', linestyle='none')
#plt.show()
#np.c_用于组合上述的维度复制，ravel函数则用于打散维度复制的数据
z = np.c_[X.ravel(), Y.ravel()]