from __future__ import absolute_import, division, print_function

import numpy as np
import numbers
import os
from scipy.special import erf, erfc
from sympy import Symbol, nsolve
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


# Function to obtain the parameters for the SELU with arbitrary fixed point (mean variance)
def get_selu_parameters(fixed_point_mean=0.0, fixed_point_var=1.0):
    """ Finding the parameters of the SELU activation function. The function returns alpha and lambda for the desired
    fixed point. """
    aa = Symbol('aa')
    ll = Symbol('ll')
    nu = fixed_point_mean
    tau = fixed_point_var
    mean = 0.5 * ll * (nu + np.exp(-nu ** 2 / (2 * tau)) * np.sqrt(2 / np.pi) * np.sqrt(tau) +
                       nu * erf(nu / (np.sqrt(2 * tau))) - aa * erfc(nu / (np.sqrt(2 * tau))) +
                       np.exp(nu + tau / 2) * aa * erfc((nu + tau) / (np.sqrt(2 * tau))))
    var = 0.5 * ll ** 2 * (np.exp(-nu ** 2 / (2 * tau)) * np.sqrt(2 / np.pi * tau) * nu + (nu ** 2 + tau) *
                           (1 + erf(nu / (np.sqrt(2 * tau)))) + aa ** 2 * erfc(nu / (np.sqrt(2 * tau))) -
                           aa ** 2 * 2 * np.exp(nu + tau / 2) * erfc((nu + tau) / (np.sqrt(2 * tau))) +
                           aa ** 2 * np.exp(2 * (nu + tau)) * erfc((nu + 2 * tau) / (np.sqrt(2 * tau)))) - mean ** 2
    eq1 = mean - nu
    eq2 = var - tau
    res = nsolve((eq2, eq1), (aa, ll), (1.67, 1.05))
    return float(res[0]), float(res[1])


print('recover the parameters of the SELU with mean zero and unit variance:')
print(get_selu_parameters(0, 1))

print('obtain new parameters for mean zero and variance 2:')
my_fixed_point_mean = -0.1
my_fixed_point_var = 2.0
my_alpha, my_lambda = get_selu_parameters(my_fixed_point_mean, my_fixed_point_var)
print((my_alpha, my_lambda))


# Adjust the SELU function and Dropout to your new parameters
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = my_alpha
        scale = my_lambda
        return scale * tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


def dropout_selu(x, rate, alpha=-my_alpha * my_lambda, fixed_point_mean=my_fixed_point_mean,
                 fixed_point_var=my_fixed_point_var, noise_shape=None, seed=None, name=None, training=False):

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

        a = tf.sqrt(fixed_point_var / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixed_point_mean, 2) +
                                                    fixed_point_var)))
        b = fixed_point_mean - a * (keep_prob * fixed_point_mean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training, lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


x = tf.Variable(tf.random_normal([10000], mean=my_fixed_point_mean, stddev=np.sqrt(my_fixed_point_var)))
w = selu(x)
y = dropout_selu(w, 0.2, training=True)
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    z, zz, zzz = sess.run([x, w, y])
    print("mean/var should be at:", my_fixed_point_mean, "/", my_fixed_point_var)
    print("Input data mean/var:  ", "{:.12f}".format(np.mean(z)), "/", "{:.12f}".format(np.var(z)))
    print("After selu:           ", "{:.12f}".format(np.mean(zz)), "/", "{:.12f}".format(np.var(zz)))
    print("After dropout mean/var", "{:.12f}".format(np.mean(zzz)), "/", "{:.12f}".format(np.var(zzz)))

# For completeness: These are the correct expressions for mean zero and unit variance
my_alpha = -np.sqrt(2/np.pi) / (np.exp(0.5) * erfc(1/np.sqrt(2))-1)
my_lambda = (1 - np.sqrt(np.exp(1)) * erfc(1 / np.sqrt(2))) * \
            np.sqrt(2 * np.pi / (2 + np.pi - 2 * np.sqrt(np.exp(1)) * (2 + np.pi) * erfc(1 / np.sqrt(2)) +
                                 np.exp(1) * np.pi * erfc(1 / np.sqrt(2)) ** 2 + 2 * np.exp(2) * erfc(np.sqrt(2))))

print("Alpha parameter of the SELU: ", my_alpha)
print("Lambda parameter of the SELU: ", my_lambda)
