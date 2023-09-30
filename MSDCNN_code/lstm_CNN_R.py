import tensorflow as tf
import numpy as np
from data_generateds1 import get_tfrecord
from MyResNet import ResNet
from main import parse_args

height, width, n_channels = 128, 128, 1
MOVING_AVERAGE_DECAY=0.9999
n_frames = 10
n_classes = 2
batch_size =128
n_hidden = 144# number of hidden cells in LSTM

def tdconv1d(inputs,k,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        in_shape=inputs.shape
        # C=in_shape[1]*in_shape[2]*in_shape[3]
        inputs=tf.reshape(inputs,[1,int(in_shape[0]),int(in_shape[1]),int(in_shape[2]),int(in_shape[3])])
        w=tf.get_variable('filter', shape=[k, 1, 1, int(in_shape[3]), int(in_shape[3])])
        # w=tf.tile(w,[1,C,C])
        out_normal=tf.nn.conv3d(inputs,w,[1,1,1,1,1],padding='SAME',name='outnormal')
        # out_normal=tf.nn.conv1d(inputs,w,1,padding="SAME")
        s1,s2,s3= tf.split(w, (1, 1,1), 0)
        diff_kernel=s1+s3
        # diff_kernel=tf.reduce_mean(w,0,keep_dims=True)
        # diff_kernel = tf.reduce_sum(w, 0, keep_dims=True)
        out_diff = tf.nn.conv3d(inputs, diff_kernel, [1, 1, 1, 1, 1], padding='SAME', name='outdiff')
        # out_diff=tf.nn.conv1d(inputs,diff_kernel,1,padding="SAME")
        out=tf.reshape(out_normal-out_diff*0.7,in_shape)
    return out
def mconv1d(inputs,C,k,name):
    out=tf.layers.conv1d(inputs,C,k,activation=tf.nn.relu,padding='valid',name=name,reuse=tf.AUTO_REUSE)
    return out
def conv_elu(inputs,C,k,name):
    out=tf.layers.conv2d(inputs,C,[k,k],[1,1],'valid',activation=tf.nn.elu,kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01),kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name,reuse=tf.AUTO_REUSE)
    return out

def conv3d_relu(inputs,filters,name):
    out=tf.layers.conv3d(inputs, filters=filters, kernel_size=[3, 3, 3], strides=1, padding='SAME',
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu,name=name,reuse=tf.AUTO_REUSE)
    return out

def maxpool3d(inputs,s,name):
    out=tf.layers.max_pooling3d(inputs, [2, 2, 2], [s, 2, 2], 'SAME', name=name)
    return out

def conv_elu_same(inputs,C,k,name):
    out=tf.layers.conv2d(inputs,C,[k,k],[1,1],'SAME',activation=tf.nn.elu,kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01),kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name,reuse=tf.AUTO_REUSE)
    return out

def pool(inputs,k,name,m='max'):
    if m =='avg':
        out=tf.layers.average_pooling2d(inputs,[k,k],[2,2],padding='SAME',name=name)
    else:
        out=tf.layers.max_pooling2d(inputs,[k,k],[2,2],padding='SAME',name=name)
    return out

def conv(inputs,C,k,name):
    out=tf.layers.conv2d(inputs,C,[k,k],[1,1],'valid',activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name,reuse=tf.AUTO_REUSE)
    return out

def fc(inputs,name):
    out=tf.layers.dense(inputs,2,activation=None,kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01),kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name,reuse=tf.AUTO_REUSE)
    return out
def fc_relu(inputs,C,name):
    out=tf.layers.dense(inputs,C,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.0005),name=name,reuse=tf.AUTO_REUSE)
    return out
def dnetbackbone(inputs):

    conv1=conv_elu(inputs,2,5,'dconv1')
    # conv1 = tdconv1d(conv1, 3, 'dtd1')
    # conv1 = tf.nn.relu(conv1)
    pool1=pool(conv1,2,'dpool1','max')



    conv2=conv_elu(pool1,4,3,'dconv2')
    # conv2 = tdconv1d(conv2, 3, 'dtd2')
    # conv2 = tf.nn.relu(conv2)
    pool2=pool(conv2,2,'dpool2','max')

    conv3=conv_elu(pool2,8,3,'dconv3')
    # conv3 = tdconv1d(conv3, 3, 'dtd3')
    # conv3 = tf.nn.relu(conv3)
    pool3 = pool(conv3, 2, 'dpool3', 'avg')

    conv4 = conv_elu(pool3, 48, 3, 'dconv4')
    # conv4 = tdconv1d(conv4, 3, 'dtd4')
    # conv4 = tf.nn.relu(conv4)
    pool4 = pool(conv4, 2, 'dpool4', 'avg')

    conv5 = conv_elu(pool4, 24, 1, 'dconv5')
    # conv5 = tdconv1d(conv5, 3, 'dtd5')
    # conv5 = tf.nn.relu(conv5)
    pool5 = pool(conv5, 2, 'dpool5', 'avg')

    conv6 = conv(pool5, 48, 3, 'dconv6')
    bn1 = tf.contrib.layers.batch_norm(inputs=conv6, decay=0.9, updates_collections=None, is_training=True)
    out=tf.nn.relu(bn1)
    return out
def netbackbone(inputs):

    conv1=conv_elu(inputs,2,5,'conv1')

    # conv1=tf.nn.elu(conv1)
    # m1 = tf.reduce_mean(conv1, 0, keep_dims=True)
    # conv1 = conv1 - m1
    # conv1=conv_elu_same(conv1,2,1,'conv1d')
    # conv1=tdconv1d(conv1,3,'td1')
    # conv1=tf.nn.elu(conv1)
    # conv1=conv1d(conv1,3,'1d1')
    # bn1 = tf.contrib.layers.batch_norm(inputs=conv1, decay=0.9, updates_collections=None, is_training=True)
    pool1=pool(conv1,2,'pool1','max')

    conv2=conv_elu(pool1,4,3,'conv2')
    # conv2 = tf.nn.elu(conv2)
    # conv2 = tdconv1d(conv2, 3, 'td2')
    # conv2=tf.nn.elu(conv2)
    # conv2 = conv1d(conv2, 3, '1d2')
    # bn2 = tf.contrib.layers.batch_norm(inputs=conv2, decay=0.9, updates_collections=None, is_training=True)
    # m2 = tf.reduce_mean(conv2, 0, keep_dims=True)
    # conv2 = conv2 - m2
    # conv2 = conv_elu_same(conv2, 4, 1, 'conv1d2')
    pool2=pool(conv2,2,'pool2','max')


    conv3=conv_elu(pool2,8,3,'conv3')
    # conv3 = tdconv1d(conv3, 3, 'td3')
    # conv3=tf.nn.elu(conv3)
    # conv3 = conv1d(conv3, 3, '1d3')
    # bn3 = tf.contrib.layers.batch_norm(inputs=conv3, decay=0.9, updates_collections=None, is_training=True)
    # m3 = tf.reduce_mean(conv3, 0, keep_dims=True)
    # conv3 = conv3- m3
    # conv3 = conv_elu_same(conv3, 8, 1, 'conv1d3')
    pool3 = pool(conv3, 2, 'pool3', 'avg')

    conv4 = conv_elu(pool3, 48, 3, 'conv4')
    # conv4 = tdconv1d(conv4, 3, 'td4')
    # conv4=tf.nn.elu(conv4)
    # conv4 = conv1d(conv4, 3, '1d4')
    # bn4 = tf.contrib.layers.batch_norm(inputs=conv4, decay=0.9, updates_collections=None, is_training=True)
    # m4 = tf.reduce_mean(conv4, 0, keep_dims=True)
    # conv4= conv4 - m4
    # conv4 = conv_elu_same(conv4, 48, 1, 'conv1d4')
    pool4 = pool(conv4, 2, 'pool4', 'avg')

    conv5 = conv_elu(pool4, 24, 1, 'conv5')
    # conv5 = tdconv1d(conv5, 3, 'td5')
    # conv5=tf.nn.elu(conv5)
    # conv5 = conv1d(conv5, 3, '1d5')
    # bn5 = tf.contrib.layers.batch_norm(inputs=conv5, decay=0.9, updates_collections=None, is_training=True)
    # m5 = tf.reduce_mean(conv5, 0, keep_dims=True)
    # conv5= conv5 - m5
    # conv5 = conv_elu_same(conv5, 24, 1, 'conv1d5')
    pool5 = pool(conv5, 2, 'pool5', 'avg')

    conv6 = conv(pool5, 48, 3, 'conv6')
    bn1 = tf.contrib.layers.batch_norm(inputs=conv6, decay=0.9, updates_collections=None, is_training=True)
    out=tf.nn.elu(bn1)

    return out

def fdiff(inputs):
    out=inputs[1:, :, :, :]-inputs[:tf.shape(inputs)[0]-1, :, :, :]
    return out
def pdiff(inputs):
    # m=tf.reduce_mean(inputs,0,keep_dims=True)
    p, _ = tf.split(inputs, (tf.shape(inputs)[0] - 1, 1), 0)
    _, q = tf.split(inputs, (1, tf.shape(inputs)[0] - 1), 0)
    out = q - p
    return out

def FetureFuse(inputs):
    instances = []
    for i in range(inputs.shape[0]):

        # con1=tf.contrib.layers.flatten(netbackbone(inputs[i, :, :, :, :]))
        con1=tf.contrib.layers.flatten(dnetbackbone(inputs[i, :, :, :, :]))
        con=pdiff(con1)
        # con=tf.contrib.layers.flatten(con)

        diff = fdiff(inputs[i, :, :, :, :])
        con2=tf.contrib.layers.flatten(netbackbone(diff))

        # con2 = tf.contrib.layers.flatten(CNN.network(diff))
        con3=tf.concat([con,con2],-1)
        instances.append(con3)
    return tf.stack(instances, axis=0)

def lstm(inputs):
    # CNN = ResNet(args)
    inputs_F=FetureFuse(inputs)
    print(inputs_F.shape)
    cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    cell1 = tf.contrib.rnn.AttentionCellWrapper(cell,2)
    # cell2=tf.contrib.rnn.DropoutWrapper(cell1,output_keep_prob=0.5)
    # cell2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # cell2 = tf.contrib.rnn.AttentionCellWrapper(cell2, 2)
    # cell3 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # cell3 = tf.contrib.rnn.AttentionCellWrapper(cell3, 2)
    # cells=[cell1,cell2,cell3]
    # mutcell=tf.contrib.rnn.MultiRNNCell(cells)

    output, _ = tf.nn.dynamic_rnn(cell1, inputs_F,
                                  initial_state=cell1.zero_state(tf.shape(inputs_F)[0], dtype=tf.float32))
    # out=[]
    # for i in range(n_frames-1):
    #     out.append(fc(output[:, i, :],'out'))
    # out=tf.reduce_mean(tf.stack(out, axis=0),0)
    # print(out.shape)
    # conv1d1=tf.contrib.layers.flatten(mconv1d(inputs_F,144,9,'conv1d'))
    # out=fc(conv1d1,"out")
    # b=inputs_F.shape[0]
    # t=inputs_F.shape[1]
    # inputs_F=tf.reshape(inputs_F,(b*t,96))
    # fc1=fc_relu(inputs_F,1024,'fc1')
    # drop1=tf.nn.dropout(fc1,0.5,name='drop1')
    # fc2=fc_relu(drop1,1024,'fc2')
    # drop2=tf.nn.dropout(fc2,0.5,name='drop2')
    # out=fc(drop2,'out')
    # out=tf.reshape(out,(b,t,2))
    # out=tf.reduce_mean(out,1)

    out=fc(output[:, -1, :],'out')
    return out

def main():
    # args = parse_args()
    X = tf.placeholder(tf.float32, shape=(batch_size, n_frames, height, width, n_channels))
    y = tf.placeholder(tf.int64, shape=(batch_size, None))
    global_step = tf.Variable(0, trainable=False)

    pred = lstm(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))# sigmoid_cross_entropy_with_logits
    optimizer = tf.train.AdamOptimizer(0.0009)
    training_op = optimizer.minimize(loss,global_step=global_step)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([training_op, ema_op]):
        train_op = tf.no_op(name='train')
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        # params = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        # print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
        sess.run(tf.global_variables_initializer())

        batch_num = 0
        X_batch, y_batch = get_tfrecord(batch_size, isTrain=True)
        X_batch = tf.reshape(X_batch, [batch_size, n_frames, height, width, n_channels])
        # ckpt = tf.train.get_checkpoint_state('./tmp/')
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        for step in range(1, 4630):#4630 59264 6650 4460
            X_batch1, y_batch1 = sess.run([X_batch, y_batch])
            batch_num += 1
            op,loss1, acc = sess.run([train_op,loss, accuracy], feed_dict={X: X_batch1, y: y_batch1})
            if step % 10 == 0:
                saver.save(sess, './tmp/after_batch_{}.ckpt'.format(batch_num))
                print('setp:%d,loss:%f,acc,%f'%(step,loss1,acc))
        saver.save(sess, './tmp/final.ckpt')
        coord.request_stop()
        coord.join(thread)

if __name__=="__main__":
    main()




