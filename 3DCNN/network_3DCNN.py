import tensorflow as tf
import numpy as np

def diff(inputs):
    input = []
    for i in range(inputs.shape[1] - 1):
        input.append(inputs[:, i+1, :, :, :] - inputs[:, i , :, :, :])
    input = tf.stack(input, axis=1)
    return input

def conv3d_relu(inputs,filters,name):
    out=tf.layers.conv3d(inputs, filters=filters, kernel_size=[3, 3, 3], strides=1, padding='SAME',
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name)
    return out

def maxpool3d(inputs,name):
    out=tf.layers.max_pooling3d(inputs, [2, 2, 2], [2, 2, 2], 'SAME', name=name)
    return out

def maxpool3df(inputs,name):
    out=tf.layers.max_pooling3d(inputs, [1, 2, 2], [1, 2, 2], 'SAME', name=name)
    return out

def maxpool3dg(inputs,name):
    out=tf.layers.max_pooling3d(inputs, [2, 1, 1], [2, 1, 1], 'SAME', name=name)
    return out

def fc(inputs,units,name):
    out=tf.layers.dense(inputs, units, activation=tf.nn.relu,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name)
    return out
def network(inputs,istrain):

    diffl=diff(inputs)

    conv1=conv3d_relu(diffl,64,'conv1')
    pool1=maxpool3df(conv1,'pool1')


    conv2=conv3d_relu(pool1,64,'conv2')
    pool2=maxpool3d(conv2,'pool2')

    conv3=conv3d_relu(pool2,64,'conv3')


    conv4 = conv3d_relu(conv3, 128, 'conv4')
    pool4 = maxpool3d(conv4, 'pool4')


    conv5 = conv3d_relu(pool4, 128, 'conv5')
    pool5 = maxpool3d(conv5, 'pool5')


    conv6 = conv3d_relu(pool5, 128, 'conv6')
    pool6 = maxpool3d(conv6, 'pool6')


    conv7 = conv3d_relu(pool6, 512, 'conv7')
    pool7 = maxpool3dg(conv7, 'pool7')


    conv8 = conv3d_relu(pool7, 512, 'conv8')
    pool8 = maxpool3dg(conv8, 'pool8')


    dense1i = tf.reshape(pool8, [pool8.get_shape().as_list()[0], np.prod(pool8.get_shape().as_list()[1:])])
    dense1 = fc(dense1i,4096,'dense1')
    dropout1 = tf.layers.dropout(dense1, 0.5, training=istrain, name='dropout1')
    dense2 = fc(dropout1,4096,'dense2')
    dropout2 = tf.layers.dropout(dense2, 0.5, training=istrain, name='dropout2')
    out = tf.layers.dense(dropout2, 2,activation=None,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),name='out')
    return out


