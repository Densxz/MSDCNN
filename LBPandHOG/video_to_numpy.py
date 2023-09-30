import os

import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def video_to_numpy1(path):
    cap = cv2.VideoCapture(path)
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))
    # print(framenum)
    video=[]
    cnt = 0
    while (cnt<framenum):
        a, b = cap.read()
        # cv2.imshow('%d' % cnt, b)
        # cv2.waitKey(20)
        # b = b.astype('float16') / 255
        # video[cnt] = b
        if b is None:
            break
        else:
            image = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            # lbp = local_binary_pattern(img, 8, 1)
            video.append(img)
    #     if cnt >= 1:
    #         video[cnt - 1] = video[cnt] - video[cnt - 1]
        cnt += 1
    # video.pop()
    print(len(video))
    # if len(video)==10:
    # return np.stack(video),len(video)#np.stack(video)
    # else:
    #     return video, len(video)
    return np.stack(video),len(video)
def video_to_numpy(path):
    cap = cv2.VideoCapture(path)
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))
    # print(framenum)
    video=[]
    cnt = 0
    while (cnt<framenum):
        a, b = cap.read()
        # cv2.imshow('%d' % cnt, b)
        # cv2.waitKey(20)
        # b = b.astype('float16') / 255
        # video[cnt] = b
        if b is None:
            break
        else:
            image = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            # lbp = local_binary_pattern(img, 8, 1)
            video.append(img)
    #     if cnt >= 1:
    #         video[cnt - 1] = video[cnt] - video[cnt - 1]
        cnt += 1
    # video.pop()
    print(len(video))
    # if len(video)==10:
    # return np.stack(video),len(video)#np.stack(video)
    # else:
    #     return video, len(video)
    return np.stack(video),len(video)
if __name__=="__main__":
    # video,n = video_to_numpy("./train/delete_IPhone_6Plus_indoor_day_without_tripod_light_08.avi")
    dir3="D:/数据集/data/train/"
    oripath=[]
    list = os.listdir(dir3)
    for i in range(0, len(list)):
        oripath=os.path.join(dir3, list[i])
        video, n = video_to_numpy1(oripath)
        if n!=10:
            print(oripath)
            os.remove(oripath)
    # cv2.imshow('huh', video[1])
    # cv2.waitKey(200)
    # image = cv2.cvtColor(video[2], cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    # img1=img.flatten()
    # print(img1)
    # # cv2.imshow('huh', img)
    # # cv2.waitKey(200)
    # lbp = local_binary_pattern(img, 8, 1)
    # #edges = filters.sobel(image)
    # cv2.imshow('huh', lbp)
    # cv2.waitKey(200000)