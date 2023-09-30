import tensorflow as tf
from network_3DCNN import network
from data_generateds1 import get_tfrecord

#训练参数
learning_rate = 0.001
training_steps = 52291#57084#85173#59264#137475
batch_size = 10
MOVING_AVERAGE_DECAY=0.99
#网络参数
height,width = 112,112
n_channels=3
n_frames = 49
num_classes = 2

X = tf.placeholder(tf.float32, [batch_size,n_frames, height, width, n_channels])
y = tf.placeholder(tf.int64, [batch_size, num_classes])
dropout=tf.placeholder(tf.bool, name='dropout')
global_step = tf.Variable(0, trainable=False)

pred=network(X,istrain=dropout)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss,global_step=global_step)
# ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# ema_op = ema.apply(tf.trainable_variables())
# with tf.control_dependencies([training_op, ema_op]):
#     train_op = tf.no_op(name='train')
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_batch, y_batch = get_tfrecord(batch_size, isTrain=True)
    X_batch1 = tf.reshape(X_batch, [batch_size, n_frames, height, width, n_channels])

    # X_batchv, y_batchv = get_tfrecord(batch_size, isTrain=False)
    # X_batchv1 = tf.reshape(X_batchv, [batch_size, n_frames, height, width, n_channels])
    # ckpt = tf.train.get_checkpoint_state('./tmp/')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    for step in range(1, training_steps):
        X_batch2, y_batch2 = sess.run([X_batch1, y_batch])
        op, loss1, acc,it,pre= sess.run([train_op, loss, accuracy,global_step,pred], feed_dict={X: X_batch2, y: y_batch2,dropout: True})
        if it % 10 == 0:
            # X_batch1v, y_batch1v = sess.run([X_batchv1, y_batchv])
            # loss1v, accv = sess.run([loss, accuracy], feed_dict={X: X_batch1v, y: y_batch1v, dropout: False})
            saver.save(sess, './tmp/after_batch_{}.ckpt'.format(it))
            # print('setp: %d,loss: %f,acc: %f,valloss: %f,valacc: %f'%(it,loss1,acc,loss1v,accv))
            print('setp: %d,loss: %f,acc: %f' % (it, loss1, acc))
    saver.save(sess, './tmp/after_batch_%d.ckpt'%(step))
    coord.request_stop()
    coord.join(thread)