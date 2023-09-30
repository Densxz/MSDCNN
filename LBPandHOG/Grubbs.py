import numpy as np
import scipy.stats as stats
from postprocessing import post
def calculate_critical_value(size, alpha):
    """Calculate the critical value with the formula given for example in
    https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers#Definition
    Args:
        ts (list or np.array): The timeseries to compute the critical value.
        alpha (float): The significance level.
    Returns:
        float: The critical value for this test.
    """
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    # print("Grubbs Critical Value: {}".format(critical_value))
    return critical_value

def Gcalculated(y):
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    index=abs_val_minus_avg.tolist().index(max_of_deviations)
    s = np.std(y)
    Gcalculated = max_of_deviations / s
    # print("Gcalculated: {}".format(Gcalculated))
    return Gcalculated,index

def gerabnormal(arr):
    rlist=arr[:]
    srlist=np.sort(rlist)
    abnormalindex=[]
    abnormals=[]
    while (len(srlist)>0) :
        G,index=Gcalculated(srlist)
        GV=calculate_critical_value(len(srlist),0.01)
        if G>GV:
            abnormalindex.append(arr.tolist().index(srlist[index]))
            abnormals.append(srlist[index])
            srlist=np.delete(srlist,[index])
        else:
            return abnormals,abnormalindex
    return abnormals,abnormalindex
# def post(arr,index):
#     dlist=arr[:]
#     dlisttemp = []
#     w = 1
#     for i in index:
#         if i < w:
#             win = dlist[0:i + w + 1]
#             win=np.delete(win,i)
#         else:
#             if i > len(dlist) - w - 1:
#                 win = dlist[i - w:]
#                 win=np.delete(win,w)
#             else:
#                 win = dlist[i - w:i + w + 1]
#                 win=np.delete(win,w)
#         winmean = np.mean(win)
#         dlist[i]=dlist[i]-winmean
#         # if i==0:
#         #     dlisttemp.append(dlist[i])
#         # else:
#         #     if dlist[i]>dlist[i-1]:
#         #         dlisttemp.append(dlist[i]-dlist[i-1])
#         #     else:
#         #         dlisttemp.append(dlist[i-1] - dlist[i])
#     # dlist = dlisttemp
#     return dlist
if __name__=='__main__':
    f=open('outputhogVISION1.txt','r')
    conetents=f.readlines()#[:60]
    f.close()
    TP,FP,TN,FN=0,0,0,0
    n=0
    for content in conetents:
        print(content.split(',')[0])

        arr=np.array(content.split(',')[1:],dtype=float)
        # dlist = post(arr)
        n=n+1
        abnormal,index=gerabnormal(arr)
        # arrpost=np.array(post(arr,index1))
        # abnormal, index = gerabnormal(arrpost)
        print(index)
        print(abnormal)
        if content.split(',')[0].split('_')[-3]!='30':
            continue
        if content.split(',')[0].split('_')[0]=='del' and index!=[] and int(content.split(',')[0].split('_')[-1][:-4])-1 in index:
            print(TP)
            TP=TP+1

        if content.split(',')[0].split('_')[0]=='del' and int(content.split(',')[0].split('_')[-1][:-4])-1 not in index:
            FN=FN+1
        if content.split(',')[0].split('_')[0]!='del' and index==[]:
            TN=TN+1
        if content.split(',')[0].split('_')[0]!='del' and index!=[]:
            FP=FP+1

    print("R=%.6f,P=%.6f,FPR=%.6f，Acc=%.6f"%(TP/(TP+FN),TP/(TP+FP),FP/(FP+TN),(TP+TN)/(TP+TN+FP+FN)))