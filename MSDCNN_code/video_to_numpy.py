import os
from dhash import calculate_hash
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def video_to_numpy1(path):
    cap = cv2.VideoCapture(path)
    wid = int(cap.get(3)) #3是返回视频流内帧的宽度
    hei = int(cap.get(4)) #4是返回高度
    framerate = int(cap.get(5)) #5返回帧速率
    framenum = int(cap.get(7)) #7返回帧数
    video=[]
    cnt = 0
    while (cnt<framenum):
        a, b = cap.read() #有读到东西a就是true，b就是返回的图片，是个三维数组
        if b is None:
            break
        else:
            image = cv2.cvtColor(b, cv2.COLOR_BGR2RGB) #将 BGR 色彩空间转换为 RGB 色彩空间
            #RGB 颜色存储在结构体或无符号整数中，其中蓝色占据最低有效区域"(32 位和 24 位格式中的一个字节)，绿色次之，红色第三次最少.
            # BGR 是相同的，只是区域的顺序颠倒了.红色占据最不重要的区域，绿色第二(静止)，蓝色第三.
            img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            img = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114 #通过彩色计算亮度的公式
            video.append(img) #给vedio列表添加入img列表
    print("img:",img)
    print("len_veido:",len(video))
    print("stack_vedio:",np.stack(video))
    return np.stack(video),len(video) #返回堆叠的video列表和列表长度

def video_to_numpy(path):
    cap = cv2.VideoCapture(path)
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))
    video=[]
    cnt = 0
    while (cnt<framenum):
        a, b = cap.read()
        if b is None:
            break
        else:
            image = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) #将 BGR 色彩空间转换为 GRAY 色彩空间
            img = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            lbp = local_binary_pattern(img, 8, 1)
            lbp = lbp.flatten()
            video.append(lbp)
        if cnt>=1:
            video[cnt-1]=video[cnt]-video[cnt-1]
        cnt += 1
    video.pop()
    print(len(video))
    return video,len(video)

if __name__=="__main__":
    dir3="D:/数据集/data/train/"
    oripath=[]
    list = os.listdir(dir3) #返回dir3路径的所有文件的列表
    for i in range(0, len(list)): #从头遍历文件
        oripath=os.path.join(dir3, list[i])
        video, n = video_to_numpy1(oripath)
        if n!=10:
            print(oripath)
            os.remove(oripath)