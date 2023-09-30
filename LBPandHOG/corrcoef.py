import math
import cv2
from chebyshev import gerabnormal
import matplotlib.pyplot as plt
from video_to_numpy import video_to_numpy1
import numpy as np
import os
winSize = (128,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9

#定义对象hog，同时输入定义的参数，剩下的默认即可
def corrcoef(F1,F2):
    w=F1.shape[0]
    h=F1.shape[1]
    fkm1=np.sum(F1)/(w*h)
    fkm2=np.sum(F2)/(w*h)
    rk=(np.sum((F1-fkm1)*(F2-fkm2)))/((np.sum((F1-fkm1)**2)*(np.sum((F2-fkm2)**2)))**0.5)
    return rk
if __name__=="__main__":
    # lbps,n=video_to_numpy1('del_D01_V_flatWA_move_0002_1914.avi')#del_D01_V_outdoorWA_panrot_0002_514.avi
    # dlist=[]
    # dklist=np.zeros((8,8))
    # for i in range(n-2):
    #
    #     rk=corrcoef(lbps[i, :, :],lbps[i+1, :, :])
    #     rk1=corrcoef(lbps[i+1, :, :],lbps[i+2, :, :])
    #     if rk>=rk1:
    #         dk=rk/rk1
    #     else:
    #         dk=rk1/rk
    #     dlist.append(dk)
    #     # if dk <-30:
    #     #     cv2.imshow('1',lbps[i,:,:])
    #     #     cv2.imshow('2',lbps[i+1,:,:])
    #     #     cv2.imshow('3',lbps[i+2,:,:])
    #     #     cv2.waitKey(0)
    #     print(i)
    # plt.plot(range(1, len(dlist) + 1), dlist)
    # # inx=int(list[j].split('_')[-1][:-4])
    # plt.plot(511, [dlist[511]], marker="o", c='r')
    # plt.plot(512, [dlist[512]], marker="o", c='g')
    # # plt.savefig('./fig/' + path[21:] + '.jpg')
    # # print(j)
    # plt.title('1')
    # plt.show()
    # plt.close()
    dir = 'D:/数据集/data/VISIONtest/'
    list = os.listdir(dir)
    f = open('./outputlbpVISION1.txt', 'w')
    for j in range(len(list)):
        path = os.path.join(dir, list[j])
        lbps, n = video_to_numpy1(path)  # del_D01_V_flatWA_move_0002_1914.avi del_D01_V_outdoorWA_panrot_0002_514.avi
        dlist = []
        f.write(list[j])
        for i in range(n - 2):
            rk = corrcoef(lbps[i, :, :], lbps[i + 1, :, :])
            rk1=corrcoef(lbps[i+1, :, :],lbps[i+2, :, :])
            if rk>=rk1:
                dk=rk/rk1
            else:
                dk=rk1/rk
            dlist.append(dk)
            f.write(',')
            f.write('%.6f' % (dk))
            # f.write(','+str(d.mean().float()))
        f.write('\n')
        f.flush()
        print(j)
    f.close()