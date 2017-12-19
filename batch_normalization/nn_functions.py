import tensorflow as tf
import numpy as np


def batch_normalization(x, out_shape, phase_train):
    """Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D Batch-Height-Width-Depth (BHWD) input maps
        out_shape:   integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('batch_norm'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_shape]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_shape]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean),
                                                                        ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def multiclass_log_loss(predicted, actual):
    row = actual.shape[0]
    col = actual.shape[1]
    sum_val = sum([actual[i, j] * np.log(max(predicted[i, j], 1.e-15)) for i in range(row) for j in range(col)])
    return -1 * sum_val / row
