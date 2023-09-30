import math
import numpy as np
from scipy import stats
# from postprocessing import post
def get_r(arr):
    m_arr = np.mean(arr)
    d_arr = abs(arr - m_arr)
    s = np.std(arr, ddof=1)  # 样本标准差，注意分母n-1
    out_ind = np.argmax(d_arr)
    return np.max(d_arr) / s, out_ind


def esd(data, alpha=0.05, max_anoms=0.10):
    n = len(data)
    r=1
    # if isinstance(max_anoms, float):
    #     r = math.ceil(max_anoms * n)
    # else:
    #     r = max_anoms
    outliers = []
    for i in range(1, r + 1):
        p = 1 - alpha / (n - i + 1) / 2
        t = stats.t.ppf(p, n - i - 1)  # p分位点
        _lambda = (n - i) * t / math.sqrt((n - i - 1 + t ** 2) * (n - i + 1))
        arr = np.delete(data, outliers)
        _r, out_ind = get_r(arr)
        if _r > _lambda:  # 超出临界值，视为异常点
            outliers.append(out_ind)
        else:
            break
    return np.delete(data, outliers), data[outliers],outliers

if __name__=='__main__':
    f=open('UCF_3DCNN1.txt','r')
    conetents=f.readlines()
    f.close()
    TP,FP,TN,FN=0,0,0,0
    n=0
    # conetents=conetents[:60]
    for content in conetents:
        print(content.split(',')[0])
        # arr=np.array(content.split(',')[1:],dtype=float)
        arr = content.split(',')[1:]
        n=n+len(arr)
        # dlist=post(arr)

        # data,outliers,index1=esd(arr,0.1)
        index1=[]
        for i in arr:
            if float(i)>0.5:
                index1.append(arr.index(i))
        # index1=arr.index(max(arr))
        # if int(content.split(',')[0].split('_')[-1][:-4])-26 == index1:
        if index1!=[] and int(content.split(',')[0].split('_')[-1][:-4])-26 in index1:
            print(TP)
            TP=TP+1
        # print(outliers)
        print(index1)
        FP = FP + len(index1)
    FP = FP - TP
    FN=len(conetents)-TP
    TN=n-TP-FP-FN
    print("R=%.6f,P=%.6f,FPR=%.6f，Acc=%.6f"%(TP/(TP+FN),TP/(TP+FP),FP/(FP+TN),(TP+TN)/(n)))