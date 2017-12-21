import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load data and reshape
mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# hyperparameters
batch_size = 128
test_size = 256
epochs = 20
learning_rate = 0.001
decay = 0.9


# define weight initializer
def init_weights(shape, stddev=0.01, name='weight'):
    return tf.Variable(tf.random_normal(shape, stddev=stddev), name=name)


# define bias initializer
def init_bias(shape, bias=0.05, name='bias'):
    return tf.Variable(tf.constant(bias, shape=shape), name=name)


# define conv2d layer
def conv2d_layer(x, w_shape, b_shape, strides, padding):
    weight = init_weights(w_shape, stddev=0.01)
    bias = init_bias(b_shape, bias=0.05)
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weight, strides, padding), bias))


# define dense layer
def dense_layer(x, in_shape, out_shape, activation=None):
    weights = init_weights([in_shape, out_shape])
    bias = init_bias([out_shape])
    y = tf.add(tf.matmul(x, weights), bias)  # y = w*x + b
    if activation is None:
        return y
    else:
        return activation(y)


with tf.Graph().as_default(), tf.Session() as sess:
    # set input data
    X = tf.placeholder(tf.float32, [None, 784])
    inputs = tf.reshape(X, [-1, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])
    p_keep_conv = tf.placeholder(tf.float32, name='keep_prob_conv')
    p_keep_hidden = tf.placeholder(tf.float32, name='keep_prob_dense')

    """Define CNN model"""
    # the first layer of conv, pooling and dropout
    l1conv = conv2d_layer(inputs, [3, 3, 1, 32], [32], strides=[1, 1, 1, 1], padding='SAME')  # out_shape=(?,28,28,32)
    l1pool = tf.nn.max_pool(l1conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # out_shape=(?,14,14,32)
    l1drop = tf.nn.dropout(l1pool, p_keep_conv)
    # the second layer of conv, pooling and dropout
    l2conv = conv2d_layer(l1drop, [3, 3, 32, 64], [64], strides=[1, 1, 1, 1], padding='SAME')  # out_shape=(?,14,14,64)
    l2pool = tf.nn.max_pool(l2conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # out_shape=(?,7,7,64)
    l2drop = tf.nn.dropout(l2pool, p_keep_conv)
    # the third layer of conv, pooling and dropout
    l3conv = conv2d_layer(l2drop, [3, 3, 64, 128], [128], strides=[1, 1, 1, 1], padding='SAME')  # out_shape=(?,7,7,128)
    l3pool = tf.nn.max_pool(l3conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # out_shape=(?,4,4,128)
    l3pool = tf.reshape(l3pool, [-1, 128 * 4 * 4])  # out_shape=(?,128*4*4)
    l3drop = tf.nn.dropout(l3pool, p_keep_conv)
    # full connect layer
    l4 = dense_layer(l3drop, 128 * 4 * 4, 625, activation=tf.nn.relu)
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    # output layer
    output = dense_layer(l4, 625, 10, activation=None)
    # weight_out = init_weights([625, 10])
    # output = tf.matmul(l4, weight_out)

    """Define operations"""
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(cost)
    predict_op = tf.argmax(output, 1)

    """Training"""
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            feed_dict = {X: trX[start:end], labels: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5}
            sess.run(train_op, feed_dict=feed_dict)
        test_indices = np.arange(len(teX))  # get a test batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        feed_dict = {X: teX[test_indices], p_keep_conv: 0.8, p_keep_hidden: 0.5}
        print(epoch, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict=feed_dict)))
