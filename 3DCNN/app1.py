# coding:utf-8
import random
import tensorflow as tf
import numpy as np
import time
import heapq
import os
from video_to_numpy import video_to_numpy1
# from data_generateds1 import *
from network_3DCNN import network
# import pandas as pd

height, width, n_channels = 112, 112, 3
downscale_factor = 8
n_frames = 49
n_classes = 2
batch_size = 1
MOVING_AVERAGE_DECAY = 0.9999


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        X = tf.placeholder(tf.float32, shape=(batch_size, n_frames, height, width, n_channels))
        # training = tf.placeholder(tf.bool)
        dropout = tf.placeholder(tf.bool, name='dropout')
        pred = network(X,istrain=dropout)
        # y = tf.nn.softmax(pred)
        preValue =tf.nn.softmax(pred)
        # preValue=tf.argmax(preValue,1)
        print(preValue.shape)
        # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()
        f1 = open('UCF_3DCNN1.txt', 'w')
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('./tmp/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                for i in range(0, len(testPicArr)):  # len(list)
                    # value=i.split(',')
                    #D:\\数据集\\data\\train\\   D:\\数据集\\NC2017_Dev_Ver1_Vid\\probe\\
                    path = os.path.join('../../data/UCFdel1/', testPicArr[i])  # list[i] value[0]
                    f1.write(testPicArr[i])
                    video, n = video_to_numpy1(path)
                    print(testPicArr[i])
                    video = np.array(video, dtype='float32').reshape((batch_size, n, height, width, n_channels))
                    vn1 = (video.shape[1] - n_frames)

                    for j in range(vn1):
                        video1 = video[:, j:j + n_frames, :, :, :]
                        video1 = np.array(video1, dtype='float32')
                        preValue1 = sess.run(preValue, feed_dict={X: video1, dropout: False})
                        # print(preValue1)
                        preValue1 = preValue1.reshape((2))
                        preValue1 = preValue1.tolist()
                        preValue1 = preValue1[1]
                        f1.write(',' + str(float(preValue1)))
                    f1.write('\n')
                    f1.flush()
                f1.close()
            else:
                print("No checkpoint file found")
                return -1


def application():
    dir ='../../data/UCFdel1/'
    list = os.listdir(dir)
    restore_model(list)



def getListMaxNumIndex(num_list, topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index = map(num_list.index, heapq.nlargest(topk, num_list))
    min_num_index = map(num_list.index, heapq.nsmallest(topk, num_list))
    f = []
    k = 0.22
    for i in max_num_index:
        l, h = max_num_index[i] - 10, max_num_index[i] + 10
        n = 0
        win = num_list[l:h]
        for j in max_num_index:
            if j in win:
                n = n + 1
        d = np.median(win) - np.min(win)
        if n < 2:
            f.append(i - k * d)
        else:
            f.append(i / n - k * d)
    return np.max(f)


def main():
    application()


if __name__ == '__main__':
    main()
