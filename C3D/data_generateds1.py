#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from video_to_numpy import*

image_train_path='D:/train/'
label_train_path='./train.txt'
tfRecord_train='../../data/UCF101/train/' #video_train.tfrecords'
image_test_path='D:/test/'
label_test_path='./test.txt'
tfRecord_test='../../data/UCF101/test/'#video_test.tfrecords'
data_path='./data'

def write_tfRecord(tfRecordName, image_path, label_path):
    # writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    n=0
    tfRecordName = tfRecordName[:-10] + '-' + str(n) + tfRecordName[-10:]
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        if num_pic%10000==0:
            writer.close()
            tfRecordName=tfRecordName[:-12]+'-'+str(n)+tfRecordName[-10:]
            writer = tf.python_io.TFRecordWriter(tfRecordName)
            n=n+1
        value = content.split(',')
        img_path = image_path + value[0]
        print(img_path)
        #img = Image.open(img_path)
        video,framenum=video_to_numpy1(img_path)
        img_raw = np.array(video,dtype='uint8').tobytes()
        labels = [0] * 2
        labels[int(value[1])] = 1  
        # framenums=[0]*2
        # framenums[1]=framenum
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                })) 
        writer.write(example.SerializeToString())
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
  
def read_tfRecord(tfRecord_path):
    # files=tf.train.match_filenames_once(tfRecord_path)
    list = os.listdir(tfRecord_path)
    files_list=[]
    for i in range(0, len(list)):
        path = os.path.join(tfRecord_path, list[i])
        files_list.append(path)
    # print(files_list)
    filename_queue = tf.train.string_input_producer(files_list, shuffle=True,num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # print(serialized_example)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label': tf.FixedLenFeature([2], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })

    label = tf.cast(features['label'], tf.float32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    #img = tf.cast(img, tf.float32)
    img = tf.reshape(img, shape=[49, 128, 128, 3])
    img = tf.cast(img, tf.float32)  # * (1. / 255)
    img = img[18:34, 0:112, 0:112, :]

    return img, label
      
def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 256,
                                                    min_after_dequeue = 10)#shuffle_batch
    # print('8888')
    # print(img_batch)
    return img_batch, label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()
