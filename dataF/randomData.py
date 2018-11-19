import os
import os.path
import numpy as np
import linecache
rootdir = "H:/deep_learning/stripe_surface/line/"
#rootdir="H:/deep_learning/ringAtListOne/deep_learning/Covnet/data/"
os.chdir(rootdir)
lines=linecache.getlines("trainDataRandom.txt")
linecache.clearcache()
index=np.random.permutation(len(lines))
new_lines=[lines[i] for i in index]
filehandle=open("trainDataRandom1.txt",'a',encoding='utf-8')
for line in new_lines:
    filehandle.write(line)
filehandle.close()