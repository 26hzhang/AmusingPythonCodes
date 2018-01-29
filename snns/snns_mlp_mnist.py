"""Tutorial on self-normalizing networks on the MNIST data set: multi-layer perceptrons
Derived from: [Aymeric Damien](https://github.com/aymericdamien/TensorFlow-Examples/)
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import os
import numbers
from sklearn.preprocessing import StandardScaler
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

mnist = input_data.read_data_sets("./data_set/MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.05
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 784  # 1st layer number of features
n_hidden_2 = 784  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropoutRate = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)


# (1) Definition of scaled exponential linear units (SELUs)
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


# (2) Definition of dropout variant for SNNs
def dropout_selu(x, rate, alpha=-1.7580993408473766, fixed_point_mean=0.0, fixed_point_var=1.0, noise_shape=None,
                 seed=None, name=None, training=False):

    """Dropout to a value with rescaling."""
    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        if tensor_util.constant_value(keep_prob) == 1:
            return x
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)
        a = tf.sqrt(fixed_point_var / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixed_point_mean, 2) + fixed_point_var)))
        b = fixed_point_mean - a * (keep_prob * fixed_point_mean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training, lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


# (3) Input data scaled to zero mean and unit variance
# (1) Scale input to zero mean and unit variance
scaler = StandardScaler().fit(mnist.train.images)
# Tensorboard
logs_path = './tmp'


# Create model
def multilayer_perceptron(x, weights, biases, rate, is_training):
    # Hidden layer with SELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # netI_1 = layer_1
    layer_1 = selu(layer_1)
    layer_1 = dropout_selu(layer_1, rate, training=is_training)
    # Hidden layer with SELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # netI_2 = layer_2
    layer_2 = selu(layer_2)
    layer_2 = dropout_selu(layer_2, rate, training=is_training)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# (4) Initialization with STDDEV of sqrt(1/n)
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=np.sqrt(1 / n_input))),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=np.sqrt(1 / n_hidden_1))),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=np.sqrt(1 / n_hidden_2)))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0))
}
# Construct model
pred = multilayer_perceptron(x, weights, biases, rate=dropoutRate, is_training=is_training)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a histogramm for weights
tf.summary.histogram("weights2", weights['h2'])
tf.summary.histogram("weights1", weights['h1'])

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = scaler.transform(batch_x)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, dropoutRate: 0.05, is_training: True})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            accTrain, costTrain, summary = sess.run([accuracy, cost, merged_summary_op],
                                                    feed_dict={x: batch_x, y: batch_y, dropoutRate: 0.0,
                                                               is_training: False})
            summary_writer.add_summary(summary, epoch)
            print("Train-Accuracy:", accTrain, "Train-Loss:", costTrain)
            batch_x_test, batch_y_test = mnist.test.next_batch(512)
            batch_x_test = scaler.transform(batch_x_test)
            accTest, costVal = sess.run([accuracy, cost], feed_dict={x: batch_x_test, y: batch_y_test,
                                                                     dropoutRate: 0.0, is_training: False})
            print("Validation-Accuracy:", accTest, "Val-Loss:", costVal, "\n")
