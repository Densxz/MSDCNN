import numpy as np
import tensorflow as tf
import os
from network_3DCNN import network
batch_size = 1
MOVING_AVERAGE_DECAY=0.9999

height,width = 112,112
n_channels=3
n_frames = 49
num_classes = 2

def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'label': tf.FixedLenFeature([2], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.float32)
    # img.set_shape([150,64,64,1])
    # img=tf.reshape(img,[16,112,112,3])
    img = tf.reshape(img, shape=[49, 128, 128, 3])
    img = tf.cast(img, tf.float32)  # * (1. / 255)
    img = img[:, 0:112, 0:112, :]
    return img,label

X = tf.placeholder(tf.float32, [batch_size,n_frames, height, width, n_channels])
y = tf.placeholder(tf.int64, [batch_size, num_classes])
dropout=tf.placeholder(tf.bool, name='dropout')
global_step = tf.Variable(0, trainable=False)

sorce=network(X,istrain=dropout)
pred=tf.nn.softmax(sorce)
write_file = open("predict.txt", "w+")
list = os.listdir('../../data/UCF101/test/')
files_list = []
for i in range(0, len(list)):
    path = os.path.join('../../data/UCF101/test/', list[i])
    files_list.append(path)
dataset = tf.data.TFRecordDataset(files_list)
dataset = dataset.map(parser)
dataset = dataset.shuffle(50).batch(batch_size=batch_size)
it = dataset.make_one_shot_iterator()
# variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
# variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./tmp/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    test_images1, test_labels1 = it.get_next()
    n,T=0,0
    while True:
        try:
            n=n+1
            test_images, test_labels = sess.run([test_images1, test_labels1])
            pre,sorce1 = sess.run([pred,sorce], feed_dict={X: test_images, dropout: False})
            predict11=np.argmax(pre,1)
            write_file.write('{},{},{},{}\n'.format(
                int(test_labels[0][1]),
                predict11[0],
                pre[0][1],
                sorce1[0][1]))
            if int(test_labels[0][1]) ==int(predict11[0]):
                T=T+1
            print(n)
        except tf.errors.OutOfRangeError:
            break
    print(T/n)

