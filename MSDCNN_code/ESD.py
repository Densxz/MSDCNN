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
    f=open('predict_lstm.txt','r')
    conetents=f.readlines()
    f.close()
    TP,FP,TN,FN=0,0,0,0
    n=0

    for content in conetents:
        #print(content.split(',')[0])#打印第一个元素
        # arr=np.array(content.split(',')[1:],dtype=float)
        yes = content.split(',')[0] #存真正的删帧情况

        arr = content.split(',')[1:] #arr存的是除第一个以外所有的元素
        n=n+len(arr) #n=n+3 即下一条记录
        print('number:',n/3)

        index1=[]
        if float(arr[1])>0.5:
            index1.append(1) #有删帧
            print("index1=",index1)
        if float(arr[1]) < 0.5:
            index1.append(0)
            print("index1=", index1)

        if index1[0]==1:#预测有删帧的情况
            if arr[0] == yes[0]:
                #在有删帧的情况下，并且前面两个都是1
                TP=TP+1 #正确识别删帧点数量
                print('TP:', TP)
            if int(yes) == 0 and int(arr[0]) == 1:
                FP=FP+1 #将非删帧点错误分类为删帧点的视频片段数量
                print('FP:', FP)

        if index1[0] == 0:#预测没删帧的情况
            if arr[0] == yes[0]:
                TN=TN+1 #正确识别非删帧点数量
                print('TN:',TN)
            if int(yes) == 1 and int(arr[0]) == 0:
                FN=FN+1 #将删帧点错误识别为非删帧点的视频片段数量
                print('FN:',FN)


    print("R=%.6f,P=%.6f,FPR=%.6f，Acc=%.6f"%(TP/(TP+FN),TP/(TP+FP),FP/(FP+TN),(TP+TN)/(n/3)))
    #召回率，精度，误报率，真确率