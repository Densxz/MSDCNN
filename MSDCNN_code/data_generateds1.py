#coding:utf-8
import tensorflow as tf
import numpy as np
from video_to_numpy import*

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_train_path=None
label_train_path=None
tfRecord_train='./dataset/VISION/train/'
image_test_path=None
label_test_path=None
tfRecord_test='./dataset/VISION/test/'

data_path=None

#创建rfrecord
def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)  
    num_pic = 0 
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split(',') #记录是否删帧的，1是有0是没有
        img_path = image_path + value[0] #视频的地址+视频名字
        print(img_path)
        video,framenum=video_to_numpy1(img_path) #引用video处理函数返回video帧列表和帧数
        img_raw = np.array(video,dtype='uint8').tobytes() #把video帧序列列表转成矩阵，必须要unit8类型的矩阵
        # tobytes() 方法可以将数组转换为一个机器值数组并返回其字节表示
        labels = [0] * 2 #在矩阵后面程一个数等于将其复制几次、【0,0】
        labels[int(value[1])] = 1  #数据集里面的label：[0,1]是有删帧的、[1,0]是没有。
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }))
        #将数据处理成二进制，方便io读取和存储
        writer.write(example.SerializeToString())
        #把tf.train.Example对象序列化为字符串，因为我们写入文件的时候不能直接处理对象，需要将其转化为字符串才能处理。
        num_pic += 1 
        print ("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")

def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print ('The directory was created successfully')
    else:
        print ('directory already exists' )
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)

  #读取数据集
def read_tfRecord(tfRecord_path):
    list = os.listdir(tfRecord_path)
    files_list = []
    for i in range(0, len(list)):
        path = os.path.join(tfRecord_path, list[i])
        files_list.append(path)
    print(files_list)
    filename_queue = tf.train.string_input_producer(files_list, shuffle=True, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    print(serialized_example)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label': tf.FixedLenFeature([2], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })

    label = tf.cast(features['label'], tf.int64)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img=tf.reshape(img,shape=[49,128,128,3])
    img = tf.cast(img, tf.float32)#* (1. / 255)p
    img=img[21:31,:,:,:]

    img=img[..., 0] * 0.212671 + img[..., 1] * 0.715160 + img[..., 2] * 0.072169#转换灰度图
    print(img)

    return img, label
      
def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train #有训练就加在训练集
    else:
        tfRecord_path = tfRecord_test #没训练就加在验证集（测试集当验证集）
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 256,
                                                    min_after_dequeue = 10)#shuffle_batch
    #随机batch输出验证集/测试集
    return img_batch, label_batch

def get_tfrecord_test(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train #有训练就加在训练集
    else:
        tfRecord_path = tfRecord_test #没训练就加在验证集（测试集当验证集）
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.batch([img, label],
                                            batch_size = num,
                                            num_threads = 2,
                                            capacity = 256)
    #顺序batch输出验证集/测试集
    return img_batch, label_batch

def main():
    read_tfRecord(tfRecord_train)

if __name__ == '__main__':
    main()
