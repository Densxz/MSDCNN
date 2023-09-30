import math
import cv2
import os
import matplotlib.pyplot as plt
from video_to_numpy import video_to_numpy1
import numpy as np
from Grubbs import gerabnormal
def corrcoef(F1,F2):
    w=F1.shape[0]
    h=1#F1.shape[1]
    fkm1=np.sum(F1)/(w*h)
    fkm2=np.sum(F2)/(w*h)
    rk=(np.sum((F1-fkm1)*(F2-fkm2)))/((np.sum((F1-fkm1)**2)*(np.sum((F2-fkm2)**2)))**0.5)
    return rk
winSize = (128,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
# lbps,n=video_to_numpy1('v7_FE.avi')
# dlist=[]
# rklist=np.zeros((8,8))
# for i in range(n-1):
#     hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
#     winStride = (8, 8)
#     hog1 = hog.compute(lbps[i, :, :], winStride).reshape((-1,))
#     hog2 = hog.compute(lbps[i+1, :, :], winStride).reshape((-1,))
#     rk=corrcoef(hog1,hog2)
#     dlist.append(rk)
# abnormal,index=gerabnormal(np.array(dlist))
# print(abnormal)
# print(index)
# plt.plot(range(1, len(dlist) + 1), dlist)
# # inx=int(list[j].split('_')[-1][:-4])
# # plt.plot(512, [dlist[512]], marker="o", c='r')
# # plt.savefig('./fig/' + path[21:] + '.jpg')
# # print(j)
# plt.title('1')
# plt.show()
# plt.close()
dir='D:/数据集/data/VISIONtest/'
list = os.listdir(dir)
f=open('./outputhogVISION1.txt','w')
for j in range(len(list)):
    path = os.path.join(dir, list[j])
    video,n=video_to_numpy1(path)#del_D01_V_flatWA_move_0002_1914.avi del_D01_V_outdoorWA_panrot_0002_514.avi
    dlist=[]
    f.write(list[j])
    for i in range(n-1):
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        winStride = (8, 8)
        hog1 = hog.compute(video[i, :, :], winStride).reshape((-1,))
        hog2 = hog.compute(video[i + 1, :, :], winStride).reshape((-1,))
        rk = corrcoef(hog1, hog2)
        dlist.append(rk)
        f.write(',')
        f.write('%.6f'%(rk))
        # f.write(','+str(d.mean().float()))
    f.write('\n')
    f.flush()
    print(j)
f.close()
