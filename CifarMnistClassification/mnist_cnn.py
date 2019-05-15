import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load data and reshape
mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
trX, trY, vaX, vaY, teX, teY = mnist.train.images, mnist.train.labels, mnist.validation.images, \
                               mnist.validation.labels, mnist.test.images, mnist.test.labels

summary_path = "./summary/summary_mnist_cnn/"
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

# hyperparameters
batch_size = 200
epochs = 10
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
    with tf.name_scope('input'):
        # set input data
        X = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        p_keep_conv = tf.placeholder(tf.float32, name='keep_prob_conv')
        p_keep_hidden = tf.placeholder(tf.float32, name='keep_prob_dense')

    with tf.name_scope("layers"):
        """Define CNN model"""
        inputs = tf.reshape(X, [-1, 28, 28, 1])
        # the first layer of conv, pooling and dropout
        l1conv = conv2d_layer(inputs, [3, 3, 1, 32], [32], strides=[1, 1, 1, 1], padding='SAME')
        l1pool = tf.nn.max_pool(l1conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l1drop = tf.nn.dropout(l1pool, p_keep_conv)
        # the second layer of conv, pooling and dropout
        l2conv = conv2d_layer(l1drop, [3, 3, 32, 64], [64], strides=[1, 1, 1, 1], padding='SAME')
        l2pool = tf.nn.max_pool(l2conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l2drop = tf.nn.dropout(l2pool, p_keep_conv)
        # the third layer of conv, pooling and dropout
        l3conv = conv2d_layer(l2drop, [3, 3, 64, 128], [128], strides=[1, 1, 1, 1], padding='SAME')
        l3pool = tf.nn.max_pool(l3conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3pool = tf.reshape(l3pool, [-1, 128 * 4 * 4])  # out_shape=(?,128*4*4)
        l3drop = tf.nn.dropout(l3pool, p_keep_conv)
        # full connect layer
        l4 = dense_layer(l3drop, 128 * 4 * 4, 625, activation=tf.nn.relu)
        l4 = tf.nn.dropout(l4, keep_prob=p_keep_hidden)
        # output layer
        output = dense_layer(l4, 625, 10, activation=None)

    """Define operations"""
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=labels))
    tf.summary.scalar("loss", cost)

    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(cost)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1)), tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summary_path + "train", sess.graph)
    test_writer = tf.summary.FileWriter(summary_path + 'test')

    """Training"""
    sess.run(tf.global_variables_initializer())
    test_acc = []
    cur_step = 0
    for epoch in range(epochs):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            cur_step += 1
            feed_dict = {X: trX[start:end], labels: trY[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0}
            _, train_loss, summ = sess.run([train_op, cost, summary], feed_dict=feed_dict)
            train_writer.add_summary(summ, cur_step)
            if cur_step % 20 == 0:
                # print("step {}, train loss: {}".format(cur_step, train_loss))
                summ1 = sess.run(summary, feed_dict={X: vaX, labels: vaY, p_keep_conv: 1.0, p_keep_hidden: 1.0})
                test_writer.add_summary(summ1, cur_step)
            if cur_step % 100 == 0:
                acc = sess.run(accuracy, feed_dict={X: teX, labels: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})
                test_acc.append(acc)
                print("Step {}, test acc: {}".format(cur_step, acc))

print(test_acc)

x = np.arange(1, len(test_acc) + 1, 1)
plt.figure(figsize=(14, 7))  # set image size
plt.plot(x, test_acc, c='r', ls='-')
plt.xlim(0, len(test_acc) + 1)
plt.ylim(0.0, 1.0)
plt.legend(['Accuracy on Test Dataset'])
plt.grid()
plt.show()
