import os
import os.path
import numpy as np
import linecache
rootdir="H:/deep_learning/ringAtListOne/n196d100/"
#rootdir="H:/deep_learning/ringAtListOne/deep_learning/Covnet/data/"
os.chdir(rootdir)
lines=linecache.getlines("testData.txt")
linecache.clearcache()
n=0
filehandle=open('testData1.txt','a',encoding='utf-8')
for line in lines:
    n+=1
    if(n%1000==500):
        filehandle.write(line)
filehandle.close()