#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from video_to_numpy import *

image_train_path='D:/数据集/data/train/'
label_train_path='./train.txt'
tfRecord_train='./data/video_train.tfrecords'
image_test_path='D:/数据集/data/test/'
label_test_path='./test.txt'
tfRecord_test='./data/video_test.tfrecords'
data_path='./data'
resize_height = 28
resize_width = 28

# def write_tfRecord(tfRecordName, image_path, label_path):
#     writer = tf.python_io.TFRecordWriter(tfRecordName)
#     num_pic = 0
#     f = open(label_path, 'r')
#     contents = f.readlines()
#     f.close()
#     for content in contents:
#         value = content.split(',')
#         img_path = image_path + value[0]
#         print(img_path)
#         #img = Image.open(img_path)
#         video,framenum=video_to_numpy1(img_path)
#
#         img_raw = video.tostring()
#         labels = [0] * 2
#         labels[int(value[1])] = 1
#         # framenums=[0]*2
#         # framenums[1]=framenum
#         example = tf.train.Example(features=tf.train.Features(feature={
#                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#                 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
#                 }))
#         writer.write(example.SerializeToString())
#         num_pic += 1
#         print ("the number of picture:", num_pic)
#     writer.close()
#     print("write tfrecord successful")

# def generate_tfRecord():
#     isExists = os.path.exists(data_path)
#     if not isExists:
#         os.makedirs(data_path)
#         print ('The directory was created successfully')
#     else:
#         print ('directory already exists' )
#     write_tfRecord(tfRecord_train, image_train_path, label_train_path)
#     write_tfRecord(tfRecord_test, image_test_path, label_test_path)
  
# def read_tfRecord(tfRecord_path):
#     filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     print(serialized_example)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                         'label': tf.FixedLenFeature([2], tf.int64),
#                                         'img_raw': tf.FixedLenFeature([], tf.string)
#                                         })
#
#     label = tf.cast(features['label'], tf.float32)
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     #img = tf.cast(img, tf.float32)
#     # img.set_shape([None,4096])
#     img=tf.reshape(img,shape=[10,128,128,1])
#     #img = tf.cast(img, tf.float32)
#     print(img)
#
#     return img, label
def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'label': tf.FixedLenFeature([2], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.float32)
    # img.set_shape([150,64,64,1])
    img=tf.reshape(img,[16,112,112,3])
    return img,label
def get_tfrecord(num):
    #files=tf.train.match_filenames_once(tfRecord_path)
    dataset=tf.data.TFRecordDataset('./data/video_test.tfrecords')
    dataset=dataset.map(parser)
    dataset=dataset.shuffle(50).batch(batch_size=num)
    dataset=dataset.repeat(50)
    it=dataset.make_one_shot_iterator()
    img_batch,label_batch=it.get_next()
    return img_batch, label_batch,it

# def main():
#     # generate_tfRecord()
#
# if __name__ == '__main__':
#     main()
