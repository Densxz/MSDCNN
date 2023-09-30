# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import heapq
import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
from video_to_numpy import video_to_numpy1
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS
def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  return images_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test(test_images):
  with tf.Graph().as_default() as tg:
      # model_name = "./sports1m_finetuning_ucf101.model"
      # test_list_file = './list/test.list'
      # num_test_videos = len(list(open(test_list_file,'r')))
      # print("Number of test videos={}".format(num_test_videos))

      # Get the sets of images and labels for training, validation, and
      images_placeholder= placeholder_inputs(FLAGS.batch_size * gpu_num)
      with tf.variable_scope('var_name') as var_scope:
        weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
                }
        biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
                'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
                }
      logits = []
      for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
          logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
          logits.append(logit)
      logits = tf.concat(logits,0)
      norm_score =logits #tf.nn.softmax(logits)
      saver = tf.train.Saver()
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      init = tf.global_variables_initializer()
      sess.run(init)
      # Create a saver for writing training checkpoints.

      ckpt = tf.train.get_checkpoint_state('./models/')
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)

      # saver.restore(sess, model_name)

      # And then after everything is built, start the training loop.
      # bufsize = 0
      # write_file = open("predict_ret.txt", "w+", bufsize)
      # write_file = open("predict_ret.txt", "w+")
      # next_start_pos = 0
      # all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
      f1 = open('VISION30_C3D.txt', 'w')
      for i in range(0, len(test_images)):  # len(list)
          # value=i.split(',')
          path = os.path.join('../../data/UCFdel1/', test_images[i])  # list[i] value[0]
          print(path)
          f1.write(test_images[i])
          testPicArr, n = video_to_numpy1(path)
          print(n)
          # print(testPicArr.shape)
          video = np.array(testPicArr, dtype='float32').reshape((1, n, c3d_model.CROP_SIZE,
                                                                 c3d_model.CROP_SIZE, 3))

          npre = []
          vn1 = (video.shape[1] - 17)

          for j in range(vn1):
              video1 = []
              video2 = video[:, j:j + 16, :, :, :]
              video1.append(video2[0])
              video1 = np.array(video1, dtype='float32')
              predict_score = norm_score.eval(
                  session=sess,
                  feed_dict={images_placeholder: video1}
              )
              preValue=predict_score[0]
              print(preValue)
              preValue = preValue.reshape((2))
              preValue = preValue.tolist()
              preValue = preValue[1]
              f1.write(',' + str(preValue))
              npre.append(preValue)
          f1.write('\n')
          f1.flush()
      f1.close()

      # predict_score = norm_score.eval(
      #           session=sess,
      #           feed_dict={images_placeholder: test_images}
      #           )
      # return predict_score[0]

def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index=[]
    max_num=heapq.nlargest(topk, num_list)
    for i in max_num:
        max_num_index.append(num_list.index(i))
    f=[]
    listc=num_list[:]
    k=0.22
    for i in max_num_index:
        l, h = i - 8, i + 8
        if l<0:
            l=0
        if h>len(num_list):
            h=len(num_list)
        n=0
        win=num_list[l:h]
        for j in max_num_index:
            if num_list[j] in win:
                n=n+1
        d=np.median(win)-np.min(win)
        if n < 2:
            f.append(num_list[i]-k*d)
            listc[i]=listc[i]-k*d
        else:
            f.append(num_list[i]/n-k*d)
            listc[i] = listc[i]/n - k * d
    return np.max(f),listc
def main(_):
    dir = '../../data/UCFdel1/'
    list = os.listdir(dir)
    # random.shuffle(list)
    # random.shuffle(list)
    # f1 = open('predict_NC2017.txt', 'w')
    # f = open('train.txt', 'r')
    # contents = f.readlines()
    # f.close()

    # f1 = open('error.txt', 'w')
    # f2 =open('train.txt','w')
    # for i in contents:
    # np_mean = np.load('./mean.npy').reshape([16, 112, 112, 3])
    run_test(list)

    # for i in range(0, len(list)):  # len(list)
    #     # value=i.split(',')
    #     path = os.path.join(dir, list[i])  # list[i] value[0]
    #     print(path)
    #     f1.write(list[i])
    #     testPicArr, n = video_to_numpy1(path)
    #     print(n)
    #     # print(testPicArr.shape)
    #     video = np.array(testPicArr,dtype='float32').reshape((1, n, c3d_model.CROP_SIZE,
    #                                c3d_model.CROP_SIZE, 3))
    #
    #     npre = []
    #     vn1 = (video.shape[1] - 17)
    #
    #     for j in range(vn1):
    #         video1 = []
    #         video2 = video[:, j:j + 16, :, :, :]
    #         video1.append(video2[0])
    #         video1=np.array(video1,dtype='float32')
    #         # for k in range(np.array(video1).shape[1]):
    #         #     print(k)
    #         #     print(video1[0][k])
    #         #     # print(video[i])
    #         #     video1[0][k] = video1[0][k] - np_mean[k]
    #         #     print(np_mean[k])
    #         #     print('888888888888888888888888')
    #         #     print(video1[0][k])
    #         preValue = run_test(video1)
    #         print(preValue)
    #         preValue = preValue.reshape((2))
    #         preValue = preValue.tolist()
    #         preValue = preValue[1]
    #         f1.write(','+str(preValue))
    #         npre.append(preValue)
    #     f1.write('\n')
    #     f1.flush()



        # score,listc=getListMaxNumIndex(npre,int(len(npre)*0.02))
        # print(score)
        # plt.plot(range(1, len(npre) + 1), npre)
        # plt.title(path[16:])
        # plt.savefig('./fig/'+path[16:]+'.jpg')
        # plt.close()
        #
        # plt.plot(range(1, len(listc) + 1), listc)
        # plt.title(path[16:])
        # plt.savefig('./fig/' + path[16:] + '1.jpg')
        # plt.close()
    # f1.close()
if __name__ == '__main__':
  tf.app.run()
