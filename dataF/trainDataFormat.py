import os
import os.path
import re

rootdir = "H:\\deep_learning\\stripe_surface\\line\\len4\\n80d50"

targetpath = "H:/deep_learning/stripe_surface/line/"
os.chdir(rootdir)

for i in range(26):
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
#            if((n-1)%1500>=1200 and (n-1)%1500<=1499):
            linelist=re.split('\s+',file_data.lstrip())
            if(float(linelist[0])>=1.3 and float(linelist[0])<=1.55):
                linelist.insert(0,'1')
                linelist.insert(0,'0')
                linelist.insert(0,'0')
                f=open(targetpath+"trainData.txt",'a',encoding='utf-8')
                f.write(','.join(linelist[0:-1]))
                f.write('\n')
                f.close()
#            elif(float(linelist[0])>=0.1 and float(linelist[0])<=0.8):
#                linelist.insert(0,'0')
#                linelist.insert(0,'1')
#                f=open("trainData.txt",'a',encoding='utf-8')
#                f.write(','.join(linelist[0:-1]))
#                f.write('\n')
#                f.close()
            file_data=""
    fileHandle.close()

