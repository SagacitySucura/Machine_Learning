import numpy as np 
import linecache
import os
import os.path
rootdir="H:/deep_learning/stripe_surface/deep_learning/Covnet/data/"
os.chdir(rootdir)

def readData(filename,labellen,datalen):
    lines=linecache.getlines(filename)
    linecache.clearcache()
    new_label=[]
    new_data=[]
    for line in lines:
        new_line=[float(i) for i in line.split(',')]
        new_label.append(new_line[0:labellen])
        new_data.append(new_line[labellen:])
    return np.array(new_label),np.array(new_data)
def generate_surface(stripe):
    stripe_surface = np.zeros([151, 151], dtype=int)
    for i in range(1,151):
        if((i-1)%(stripe*2)<stripe):
            stripe_surface[i,:] = 1
        else:
            stripe_surface[i,:] = 0
    return stripe_surface
def generate_real(stripe,conformation):
    nsize = [144,144,144,144,140]
    index = int(stripe/2-1)
    real = conformation.copy()
    flag1 = False
    flag2 = False
    for i in range(196):
        if(real[i,0]>120):
            flag1=True
        if(real[i,0]==1):
            flag2=True
    if(flag1 and flag2):
        for i in range(196):
            if(real[i,0]>70):
                real[i,0]-=nsize[index]
    flag1 = False
    flag2 = False
    for i in range(196):
        if(real[i,1]>120):
            flag1=True
        if(real[i,1]==1):
            flag2=True
    if(flag1 and flag2):
        for i in range(196):
            if(real[i,1]>70):
                real[i,1]-=nsize[index]
    return real

def judge(stripe, conformation):
    stripe_surface = generate_surface(stripe)
    real = generate_real(stripe,conformation)
    flag = False
    firstM = 0
    index = 0
    for i in range(196):
        z = int(conformation[i,2])
        x = int(conformation[i,0])
        y = int(conformation[i,1])
        if(z==1 and stripe_surface[x, y]==1):
            firstM = i
            break
    tag = 0
    for i in range(1,196):
        index = (i+firstM)%196 
        x = int(conformation[index,0])
        y = int(conformation[index,1])
        z = int(conformation[index,2])
        if(z==1 and stripe_surface[x,y]==1):
            present = real[index,0]
            before = real[(index-tag-1+196)%196,0]
            dis = abs(present - before)
            if(dis>stripe):
                flag = True
                break
            tag = 0
        else:
            tag = tag + 1   
    return flag
def adsorption(stripe,conformation):
    stripe_surface = generate_surface(stripe)
    flag = False
    for i in range(196):
        x = int(conformation[i,0])
        y = int(conformation[i,1])
        z = int(conformation[i,2])
        if(z == 1 and stripe_surface[x,y]==1):
            flag=True
            break
    return flag
def calEnergy(stripe,conformation,Energys,j):
    stripe_surface = generate_surface(stripe)
    for i in range(196):
        x = int(conformation[i,0])
        y = int(conformation[i,1])
        z = int(conformation[i,2])
        if(z == 1 and stripe_surface[x,y]==1):
            Energys[int(stripe/2-1),j]+=1


test_size = len(os.listdir("testColloction/"))
#result行代表条纹宽度序号，分别是2，4，6，8
result = np.zeros([4, 2], dtype=float)
Energys = np.zeros([4,2],dtype=float)
for i in range(test_size):
    labelT,dataT=readData('testColloction/test'+str(i)+'.txt',2,588)
    try:
        label = labelT[:,0]
    except:
        print(np.shape(label))
    else:
        for j in range(len(label)):
            conformation = np.reshape(dataT[j], [196, 3])
            stripe = int(label[j])
            if(judge(stripe, conformation)):
                result[int(stripe/2-1),0] +=1
                calEnergy(stripe,conformation,Energys,0)
            elif(adsorption(stripe,conformation)):
                result[int(stripe/2-1),1] +=1
                calEnergy(stripe,conformation,Energys,1)
            else:
                if(int(label[j])==6):
                    fi=open('666.txt','a',encoding='utf-8')
                    fi.write(str(conformation))
                    fi.write('\n**********************\n')
                    fi.close()

f = open('result.txt','a',encoding='utf-8')
for i in range(len(result)):
    Energys[i,0] = Energys[i,0]/result[i,0]
    Energys[i,1] = Energys[i,1]/result[i,1]
    stripe_len = str((i+1)*2)
    f.write(stripe_len+'\t'+str(Energys[i,0])+'\t'+str(Energys[i,1])+'\n')
    f.write(stripe_len+'\t'+str(result[i,0])+'\t'+str(result[i,1])+'\n')
f.close()


