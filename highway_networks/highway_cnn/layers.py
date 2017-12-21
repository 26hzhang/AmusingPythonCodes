import tensorflow as tf
from highway_networks.highway_full_connect.layers import init_weight, init_bias


def conv2d_layer(x, w_shape, b_shape, strides, padding):
    """
    This function is used to define a normal conv2d layer
    :param x: input data
    :param w_shape: weight shape
    :param b_shape: bias shape
    :param strides: strides
    :param padding: padding
    :return: normal conv2d output
    """
    weight = init_weight(w_shape, stddev=0.1)
    bias = init_bias(b_shape, bias=0.1)
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weight, strides, padding), bias))


def conv2d_highway_layer(x, w_shape, b_shape, strides, padding, carry_bias=-1.0):
    """
    This function is used to define 2-D highway layer for convolutional networks
    :param x: input data
    :param w_shape: weight shape
    :param b_shape: bias shape
    :param strides: strides
    :param padding: padding
    :param carry_bias: bias for carry gate
    :return: highway conv2d output
    """
    weight = init_weight(w_shape, stddev=0.1)
    bias = init_bias(b_shape, bias=0.1)
    trans_weight = init_weight(w_shape, stddev=0.1)
    trans_bias = init_bias(b_shape, bias=carry_bias)
    hx = tf.nn.relu(tf.add(tf.nn.conv2d(x, weight, strides, padding), bias), name='activation')
    t = tf.sigmoid(tf.add(tf.nn.conv2d(x, trans_weight, strides, padding), trans_bias), name="transform_gate")
    c = tf.subtract(1.0, t, name="carry_gate")
    return tf.add(tf.multiply(hx, t), tf.multiply(x, c), name="highway_output")
