import tensorflow as tf
from .nn_layers import Conv2dLayer, MaxPooling2dLayer, DenseLayer, SoftmaxLayer
from .nn_functions import batch_normalization


def batch_norm_cnn(x, y_, keep_prob, phase_train):
    with tf.variable_scope('conv_1'):
        conv1 = Conv2dLayer(x, [5, 5, 1, 32], [32], strides=[1, 1, 1, 1], padding='SAME', activation=None)
        conv1_bn = batch_normalization(conv1.fit(), 32, phase_train)
        pool1 = MaxPooling2dLayer(tf.nn.relu(conv1_bn), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        pool1_out = pool1.fit()

    with tf.variable_scope('conv_2'):
        conv2 = Conv2dLayer(pool1_out, [5, 5, 32, 64], [64], strides=[1, 1, 1, 1], padding='SAME', activation=None)
        conv2_bn = batch_normalization(conv2.fit(), 64, phase_train)
        pool2 = MaxPooling2dLayer(tf.nn.relu(conv2_bn), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        pool2_flat = tf.reshape(pool2.fit(), [-1, 7 * 7 * 64])

    with tf.variable_scope('full_connected'):
        fc = DenseLayer(pool2_flat, 7 * 7 * 64, 1024)
        fc_dropped = tf.nn.dropout(fc.fit(), keep_prob)

    with tf.variable_scope('softmax_output'):
        predict = SoftmaxLayer(fc_dropped, 1024, 10).fit()

    with tf.variable_scope('loss'):
        cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(predict), reduction_indices=[1]))
        tf.summary.scalar('loss', cost)

    with tf.variable_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(y_, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return cost, accuracy, predict
