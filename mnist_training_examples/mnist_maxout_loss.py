import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)  # load data

learning_rate = 0.01  # set learning rate
epochs = 250  # set training epoches
batch_size = 100  # set batch size

# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='inputs')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='labels')


# define maxout
def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
                         .format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


# define weight variable initializer
def init_weight(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name=name)


# define bias variable initializer
def init_bias(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


def make_predictions():
    w1 = init_weight('weight_1', [784, 100])
    b1 = init_bias('bias_1', [100])
    w2 = init_weight('weight_2', [50, 10])
    b2 = init_bias('bias_2', [10])
    t = max_out(tf.matmul(x, w1) + b1, 50)
    return tf.nn.softmax(tf.matmul(t, w2) + b2)


# Construct model and encapsulating all ops into scopes
with tf.name_scope('predict'):
    pred = make_predictions()

with tf.name_scope('loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

with tf.name_scope('sgd'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

with tf.name_scope('test_loss'):
    test_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

train_loss = tf.summary.scalar('loss', cost)
accuracy = tf.summary.scalar('accuracy', accuracy)
test_loss = tf.summary.scalar('test_loss', test_loss)

train_merged = tf.summary.merge([train_loss, accuracy])
test_merged = tf.summary.merge([test_loss])

# 1. using tensorboard to visualize (go to terminal: $ tensorboard --logdir=<path_to>/tensorflow_logs)
# 2. plot with matplotlib
train_loss_list = []
test_loss_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initializing the variables
    # write logs to Tensorboard
    train_writer = tf.summary.FileWriter('./data/tensorflow_logs/example/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter('./data/tensorflow_logs/example/test')
    # Training cycle
    for epoch in range(epochs):
        train_avg_loss = 0.0
        total_batches = int(mnist.train.num_examples / batch_size)
        # loop over all batches
        for i in range(total_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, train_summary = sess.run([optimizer, cost, train_merged], feed_dict={x: batch_xs, y: batch_ys})
            train_writer.add_summary(train_summary, epoch * total_batches + i)
            train_avg_loss += c / total_batches  # Compute average loss
        train_loss_list.append(train_avg_loss)
        test_avg_loss, _, test_summary = sess.run([cost, test_loss, test_merged], feed_dict={x: mnist.test.images,
                                                                                             y: mnist.test.labels})
        test_writer.add_summary(test_summary, epoch * total_batches)
        test_loss_list.append(test_avg_loss)
        print('Epoch: %4d | training cost=%9.6f | test cost=%9.6f' % ((epoch + 1), train_avg_loss, test_avg_loss))

# plot
train_loss_list = np.array(train_loss_list)
test_loss_list = np.array(test_loss_list)
x = np.arange(0, 250, 1)
plt.figure(figsize=(14, 7))  # set image size
plt.plot(x, train_loss_list, c='r', ls='dotted')
plt.plot(x, test_loss_list, c='g')
plt.xlim(0, 260)
plt.ylim(0, 0.8)
plt.legend(['training loss', 'test loss'])
plt.show()
