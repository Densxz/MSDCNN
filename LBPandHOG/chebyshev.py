import numpy as np

def chebyshev(sta,k):
    aver=np.mean(sta)
    diff=np.abs(sta-aver)
    signa = np.std(sta)
    index=np.where(diff < k * signa)
    return index
def chebyshev2(sta1,sta2,k):
    aver=np.mean(sta2)
    diff=np.abs(sta1-aver)
    signa = np.std(sta2)
    index=np.where(diff >= k * signa)
    return index
def chebyshev3(sta,k):
    aver=np.mean(sta)
    diff=np.abs(sta-aver)
    signa = np.std(sta)
    index=np.where(diff > k * signa)
    return index
def gerabnormal(arr):
    rlist = arr[:]
    # srlist = np.sort(rlist)
    index1=chebyshev(rlist, 2)
    newsrlist=rlist[index1]
    index2 = chebyshev2(rlist,newsrlist, 6)
    # abnormal=rlist[index2]
    index=[]
    for i in index2:
        for j in i:
            index.append(j)
    findex=getanpair(rlist,index)
    abnormal=rlist[findex]
    return abnormal,findex
def getanpair(sta,index1):
    index=[]
    for i in range(len(index1)-1):
        if index1[i]==(index1[i+1]-1):
            index.append([index1[i],index1[i+1]])
    # print(index)
    if len(index)==0:
        return index
    if len(index)==1:
        return [index[0][1]]
    abmean=[]
    for j in range(len(index)):
        abmean.append((sta[index[j][0]]+sta[index[j][1]])/2)
    abmmean=np.mean(abmean)
    abindex=[]
    # print(abmean)
    # print(abmmean)
    for i in range(len(abmean)):
        if abmean[i]>=1*abmmean:
            abindex.append(i)
    finaindex=[]
    for i in range(len(abindex)):
        # finaindex.append(index[abindex[i]*2+1])
        finaindex.append(index[abindex[i]][1])
    return finaindex
if __name__=='__main__':
    f=open('outputlbpUCF1.txt','r')
    conetents=f.readlines()
    f.close()
    TP,FP,TN,FN=0,0,0,0
    n=0
    for content in conetents:
        print(content.split(',')[0])
        arr=np.array(content.split(',')[1:],dtype=float)
        n=n+1
        abnormal,index=gerabnormal(arr)
        # if len(index)==1:
        if content.split(',')[0].split('_')[-3]!='30':
            continue
        if content.split(',')[0].split('_')[0] == 'del' and index != [] and int(
                content.split(',')[0].split('_')[-1][:-4]) - 1 in index:
            print(TP)
            TP = TP + 1
        if content.split(',')[0].split('_')[0] == 'del' and int(
                content.split(',')[0].split('_')[-1][:-4]) - 1 not in index:
            FN = FN + 1
        if content.split(',')[0].split('_')[0] != 'del' and index == []:
            TN = TN + 1
        if content.split(',')[0].split('_')[0] != 'del' and index != []:
            FP = FP + 1
        print(index)
        print(arr[index])
    print("R=%.6f,P=%.6f,FPR=%.6f,Acc=%.6f"%(TP/(TP+FN),TP/(TP+FP),FP/(FP+TN),(TP+TN)/(TP+TN+FP+FN)))