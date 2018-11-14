import os
import os.path
import numpy as np
import linecache
rootdir="H:/deep_learning/ringAtListOne/deep_learning/allconnectnet/coord_data/"
#rootdir="H:/deep_learning/ringAtListOne/deep_learning/Covnet/data/"
os.chdir(rootdir)
#list[m:n]取数据的规则是包含左边而不包含右边
def readData(filename,labellen,datalen):
    lines=linecache.getlines(filename)
    linecache.clearcache()
    new_label=[]
    new_data=[]
    for line in lines:
        new_line=[float(i) for i in line.split(',')]
        new_label.append(new_line[0:labellen])
        new_data.append(new_line[(labellen+1):])
    return np.array(new_label),np.array(new_data)

label,data=readData('train1.txt',2,588)

print(data)