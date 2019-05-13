import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from .layers import dense_layer, highway_layer
import time


mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)  # load mnist data for training and testing

input_shape = 784  # 28x28x1, number of pixels for a MNIST image
output_shape = 10  # number of classes for MNIST dataset
hidden_layer_size = 50  # number of neurons for hidden layer
number_of_layers = 18  # the layers for a network
carry_bias = -20.0  # cary bias used at transform gate inside highway layer
learning_rate = 0.01  # learning rate for training
batch_size = 64  # mini-batch size
epochs = 40  # train dataset 40 times

inputs = tf.placeholder(tf.float32, [None, input_shape], name="input")  # define inputs for tensorflow graph
targets = tf.placeholder(tf.float32, [None, output_shape], name="output")  # define outputs for tensorflow graph

# define a highway networks
prev_layer = None
output_layer = None
for layer_index in range(number_of_layers):
    if layer_index == 0:
        prev_layer = dense_layer(inputs, input_shape, hidden_layer_size)
    elif layer_index == number_of_layers - 1:
        output_layer = dense_layer(prev_layer, hidden_layer_size, output_shape, activation=None)
    else:
        prev_layer = highway_layer(prev_layer, hidden_layer_size, carry_bias=carry_bias)

# define cost
with tf.name_scope('loss'):
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=targets))

# define optimizer
with tf.name_scope('sgd'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# compute accuracy
with tf.name_scope('accuracy'):
    y_pred = tf.argmax(tf.nn.softmax(output_layer), 1)
    y_true = tf.argmax(targets, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

with tf.Session() as sess:
    # initialize all parameters
    sess.run(tf.global_variables_initializer())
    # training
    for epoch in range(epochs):
        train_cost = []
        train_time = time.time()
        for batch_index in range(mnist.train.num_examples // batch_size):
            batch_imgs, batch_labels = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={inputs: batch_imgs, targets: batch_labels})
            train_cost.append(c)
        print("Epoch: {}/{}".format(epoch + 1, epochs), "  |  Current loss: {:9.6f}".format(np.mean(train_cost)),
              "  |  Epoch time: {:5.2f}s".format(time.time() - train_time))
        print("Test accuracy %g" % sess.run(accuracy, feed_dict={inputs: mnist.test.images,
                                                                 targets: mnist.test.labels}))
    # Testing
    print("Test Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, targets: mnist.test.labels}))
    # Validation
    print("Validation Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.validation.images,
                                                                 targets: mnist.validation.labels}))
