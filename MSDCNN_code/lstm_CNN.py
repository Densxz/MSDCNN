import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops #相当于C的bool？函数1：函数2，对就返回函数1，错就返回2
from tensorflow.python.training import moving_averages
from datetime import datetime
from data_generateds1 import *
# for Tensorboard logging and visualization

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
height, width, n_channels = 128, 128, 1
downscale_factor = 8
MOVING_AVERAGE_DECAY=0.9999
n_frames = 10
n_classes = 2 #输出图的通道数，也就是最终得到几张特征图
batch_size =8
n_hidden = 144# number of hidden cells in LSTM
layers = [#(5, 5, 3),
            # ('max', (1, 2, 2, 1), (1, 2, 2, 1)),
            (5, 5, 2),
            ('max', (1, 2, 2, 1), (1, 2, 2, 1)),
            (3, 3, 4),
            ('max', (1, 2, 2, 1), (1, 2, 2, 1)),
            (3, 3, 8),
            ('avg', (1, 2, 2, 1), (1, 2, 2, 1)),
            (3, 3, 48),
            ('avg', (1, 2, 2, 1), (1, 2, 2, 1)),
            (1, 1, 24),
            ('avg', (1, 2, 2, 1), (1, 2, 2, 1)),
            (3, 3, 48),
          ]

def conv_pool(x,layers):
    out = x
    n_conv, n_pool = 0, 0
    prev_depth = int(x.shape[3])
    for l in layers:
        if type(l[0]) == int:
            n_conv += 1
            with tf.variable_scope('conv_{}'.format(n_conv), reuse=tf.AUTO_REUSE):
                #用于创建名为“conv_n_conv'”的变量作用域，其中“n_conv”是用于为作用域创建唯一名称的变量。
                #“reuse=tf.AUTO_reuse”参数用于重用作用域中存在的变量（如果存在），或创建新的变量（不存在）。
                w = tf.get_variable('filter',
                                    initializer=tf.truncated_normal((l[0], l[1], prev_depth, l[2]), 0, 0.1))
                b = tf.get_variable('bias', initializer=tf.zeros(l[2]))
                con=tf.nn.conv2d(out, w, strides=(1, 1, 1, 1), padding='VALID') + b#VALID
                if n_conv==6:
                    con = tf.contrib.layers.batch_norm(inputs=con,
                                                       decay=0.9,
                                                       updates_collections=None,#这是关键图的
                                                       is_training=True)
                                                       #variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES)
                    #bn层
                out=tf.nn.elu(con)
            prev_depth = l[2]
        elif l[0] == 'max':
            n_pool += 1
            out = tf.nn.max_pool(out, l[1], l[2], padding='SAME', name='pool_{}'.format(n_pool))
        elif l[0] == 'avg':
            n_pool += 1
            out = tf.nn.avg_pool(out, l[1], l[2], padding='SAME', name='pool_{}'.format(n_pool))
    return out

def get_feature_maps(x):
    instances = []
    for i in range(x.shape[0]): #读取矩阵第一维度的长度，即数组的行数。
        con1=tf.contrib.layers.flatten(conv_pool(x[i, :, :, :, :], layers)) #先卷积后拆分（卷积部分）

        xt=x[i, 1:, :, :, :]-x[i, :tf.shape(x)[1]-1, :, :, :]#差分
        con2=tf.contrib.layers.flatten(conv_pool(xt, layers))#再卷积

        # 先卷积后差分（差分部分）
        p,_=tf.split(con1,(tf.shape(x)[1]-1,1),0)
        _,q=tf.split(con1,(1,tf.shape(x)[1]-1),0)
        con=q-p

        # 两个尺度合体
        con3=tf.concat([con,con2],-1)
        instances.append(con3)

    return tf.stack(instances, axis=0)#将instances列表中的张量沿着第0个维度进行堆叠，生成一个新的张量。

def lstm(X):
    X_features = get_feature_maps(X) #卷积完成了，并且两个差分完后，得到了特征图
    print(X_features.shape)

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    cell1=tf.contrib.rnn.AttentionCellWrapper(cell,2)
    #创建一个基本LSTM单元，该单元具有n_hidden隐藏单元和1.0的遗忘偏差。
    #用注意力包裹这个单元格，创建一个实现注意力的新单元格。注意力长度为2，这意味着注意力机制在计算注意力权重时会考虑前两个时间步长。

    output, _ = tf.nn.dynamic_rnn(cell1, X_features,
                                  initial_state=cell1.zero_state(tf.shape(X_features)[0], dtype=tf.float32))
    #dynamic_rnn函数用于创建一个可以处理可变长度序列的rnn单元。
    #cell1对象是RNN单元格的实例，tf.Tensor对象X_feature是RNN单元的输入
    #tf.shape（X_features）[0]返回X_ features张量中的行数

    with tf.variable_scope('out', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weight', initializer=tf.truncated_normal((n_hidden, n_classes), 0, 0.1))
        b = tf.get_variable('bias', initializer=tf.zeros(n_classes))
        pred = tf.matmul(output[:, -1, :], w) + b#output[:, -1, :]
    return pred

def main():
    X = tf.placeholder(tf.float32, shape=(batch_size, n_frames, height, width, n_channels))
    y = tf.placeholder(tf.int64, shape=(batch_size, None))

    global_step = tf.Variable(0, trainable=False)
    #trainable：如果为True，则会默认将变量添加到图形集合GraphKeys.TRAINABLE_VARIABLES中。
    #此集合用于优化器Optimizer类优化的的默认变量列表【可为optimizer指定其他的变量集合】，可就是要训练的变量列表。
    pred = lstm(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    #loss计算pred和y之间交叉熵的损失，tf.reduce_mean函数计算批次中所有示例的张量平均值，得出每个示例的平均损失
    optimizer = tf.train.AdamOptimizer(0.0009)
    training_op = optimizer.minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([training_op, ema_op]):
        train_op = tf.no_op(name='train')

    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
    #如果预测输出等于实际输出，则correct_pred变量将设置为True，否则将设置为False
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('acc', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter(logdir, sess.graph) #创建写入文档
        sess.run(tf.global_variables_initializer())

        batch_num = 0
        X_batch, y_batch = get_tfrecord(batch_size, isTrain=True)
        X_batchv, y_batchv = get_tfrecord(batch_size, isTrain=False)
        X_batch = tf.reshape(X_batch, [batch_size, n_frames, height, width, n_channels])
        X_batchv = tf.reshape(X_batchv, [batch_size, n_frames, height, width, n_channels])
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        for step in range(1, 59264):#6655):137475
            X_batch1, y_batch1 = sess.run([X_batch, y_batch])
            batch_num += 1
            op= sess.run([train_op], feed_dict={X: X_batch1, y: y_batch1})

            if step % 10 == 0:
                X_batch1v, y_batch1v = sess.run([X_batchv, y_batchv])
                summary_str = merged_summary_op.eval(feed_dict={X: X_batch1v, y: y_batch1v})  # ,training:True

                file_writer.add_summary(summary_str, batch_num)

                loss1, acc = sess.run([loss, accuracy], feed_dict={X: X_batch1, y: y_batch1})
                loss1v, accv = sess.run([loss, accuracy], feed_dict={X: X_batch1v, y: y_batch1v})
                saver.save(sess, './tmp/after_batch_{}.ckpt'.format(batch_num))
                print('step:%d, loss: %f, acc: %f,vloss: %f, vacc: %f'%(step,loss1,acc,loss1v,accv))
        saver.save(sess, './tmp/final.ckpt')
        coord.request_stop()
        coord.join(thread)
    file_writer.close()


def _concact_features(conv_output):
    """
    对特征图进行reshape拼接
    :param conv_output:输入多通道的特征图
    :return:
    """
    num_or_size_splits = conv_output.get_shape().as_list()[-1]
    each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
    concact_size = int(np.math.sqrt(num_or_size_splits) / 1)
    all_concact = None
    for i in range(concact_size):
        row_concact = each_convs[i * concact_size]
        for j in range(int(num_or_size_splits/concact_size)-1):
            row_concact = tf.concat([row_concact, each_convs[i * concact_size + j + 1]], 1)
        if i == 0:
            all_concact = row_concact
        else:
            all_concact = tf.concat([all_concact, row_concact], 2)

    return all_concact

# a list that specifies convolution-pooling architecture;
# list index indicate layer position in stack;
# a pooling layer is represented by a tuple: (pooling type, kernel_size, strides)
# a convolution layer is represented by a typle: (filter_height, filter_width, depth)

if __name__=='__main__':
    main()
