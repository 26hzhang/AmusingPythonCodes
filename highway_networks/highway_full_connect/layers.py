import tensorflow as tf


def dense_layer(x, in_shape, out_shape, activation=tf.nn.relu):
    """
    This function performs the a full connected layer in a network
    :param x: input for a dense layer
    :param in_shape: number of neurons the input has
    :param out_shape: the number of neurons the layer output
    :param activation: activation function
    :return: a full connected dense layer with parameters
    """
    weights = init_weight([in_shape, out_shape])
    bias = init_bias([out_shape])
    y = tf.add(tf.matmul(x, weights), bias)  # y = w*x + b
    if activation is not None:
        return activation(y)
    else:
        return y


def highway_layer(x, layer_size, carry_bias=-2.0, activation=tf.nn.relu):
    """
    This function is used to define a full connected highway network layer
        reference paper: <https://arxiv.org/pdf/1505.00387.pdf>
    :param x: input data
    :param layer_size: number of neurons in highway layer
    :param carry_bias: value for the carry bias used in transform gate
    :param activation: activation function
    :return: a full connected highway layer with parameters
    """
    # define weight and bias for activation gate
    weight = init_weight([layer_size, layer_size])
    bias = init_bias([layer_size])
    # define weight and bias for transform gate (carry gate is 1 - transform gate)
    trans_weight = init_weight([layer_size, layer_size])
    trans_bias = init_bias([layer_size], bias=carry_bias)
    # compute activation output
    hx = activation(tf.add(tf.matmul(x, weight), bias), name="input_gate")
    # compute transform gate and carry gate
    t = tf.nn.sigmoid(tf.add(tf.matmul(x, trans_weight), trans_bias), name="transform_gate")
    c = tf.subtract(1.0, t, name='carry_gate')
    return tf.add(tf.multiply(hx, t), tf.multiply(x, c), name="highway_output")


def init_weight(shape, stddev=0.05, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def init_bias(shape, bias=0.05, name="weight"):
    return tf.Variable(tf.constant(bias, shape=shape), name=name)
