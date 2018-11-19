import os
import os.path
import re
#rootdir = "H:/deep_learning/stripe_surface/AtListOne/6/n196d50/"
rootdir = "H:\\deep_learning\\stripe_surface\\line\\len4\\n80d50"
targetpath = "H:/deep_learning/stripe_surface/line/"
os.chdir(rootdir)

for i in range(26, 32):
    filename='bondarycoord'+str(i)+'.txt'
    fileHandle=open(filename,'r',encoding='utf-8')
    n=0
    file_data=""
    for line in fileHandle.readlines():
        n=n+1
        if(n%2!=0):
            line=line.strip('\n')
        file_data+=line
        if(n%2==0):
            linelist=re.split('\s+',file_data.lstrip())
            f=open(targetpath+"testData.txt",'a',encoding='utf-8')
            f.write(','.join(linelist[0:-1]))
            f.write('\n')
            f.close()
            file_data=""
    fileHandle.close()

