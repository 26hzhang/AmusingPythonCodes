from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


class Conv2dLayer(object):
    def __init__(self, inputs, w_shape, b_shape, strides, padding, activation=tf.nn.relu):
        self.input = inputs
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.weight = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), trainable=True)
        self.bias = tf.Variable(tf.constant(0.1, shape=b_shape), trainable=True)
        self.params = [self.weight, self.bias]
        self.output = None

    def fit(self):
        if self.activation is not None:
            self.output = self.activation(tf.add(tf.nn.conv2d(self.input, self.weight, strides=self.strides,
                                                              padding=self.padding), self.bias))
            return self.output
        else:
            self.output = tf.add(tf.nn.conv2d(self.input, self.weight, strides=self.strides, padding=self.padding),
                                 self.bias)
            return self.output


class MaxPooling2dLayer(object):
    def __init__(self, inputs, ksize, strides):
        self.input = inputs
        self.ksize = ksize
        self.strides = strides
        self.output = None

    def fit(self):
        self.output = tf.nn.max_pool(self.input, ksize=self.ksize, strides=self.strides, padding='SAME')
        return self.output


class DenseLayer(object):
    def __init__(self, inputs, in_shape, out_shape):
        self.input = inputs
        self.weight = tf.Variable(tf.truncated_normal([in_shape, out_shape], mean=0.0, stddev=0.05), trainable=True)
        self.bias = tf.Variable(tf.zeros([out_shape]), trainable=True)
        self.params = [self.weight, self.bias]
        self.output = None

    def fit(self):
        self.output = tf.nn.relu(tf.add(tf.matmul(self.input, self.weight), self.bias))
        return self.output


class SoftmaxLayer(object):
    def __init__(self, inputs, in_shape, out_shape):
        self.input = inputs
        self.w = tf.Variable(tf.random_normal([in_shape, out_shape], mean=0.0, stddev=0.05), trainable=True)
        self.b = tf.Variable(tf.zeros([out_shape]), trainable=True)
        self.params = [self.w, self.b]
        self.output = None

    def fit(self):
        self.output = tf.nn.softmax(tf.add(tf.matmul(self.input, self.w), self.b))
        return self.output
