# encoding: utf-8
from kmeans import KMeans
import tensorflow as tf
import numpy as np

"""
RBF Networks
"""


class RBFNet(object):
    def __init__(self, k=20, delta=0.1):
        """
        Params:
            _delta: rbf gaussian kernel
            _weights: weights from hidden layer to output layer
            _input_n: number of input size
            _hidden_num: number of hidden layer neurons
            _output_n: number of output sizes
        """
        self._delta = delta
        self._weights = None
        self._input_num = 0
        self._hidden_num = k
        self._output_num = 0
        self.sess = tf.Session()

    def setup(self, n_in, n_hidden, n_out):
        """build network"""
        self._input_num = n_in
        self._hidden_num = n_hidden
        self._output_num = n_out
        self.input_layer = tf.placeholder(tf.float32, [None, self._input_num], name='inputs_layer')
        self.output_layer = tf.placeholder(tf.float32, [None, self._output_num], name='outputs_layer')
        self.cal_centroid_vectors(self._inputs)
        self.hidden_centers = tf.constant(self.hidden_centers, name="hidden")
        self.hidden_layer = self.rbfunction(self.input_layer, self.hidden_centers)

    def fit(self, inputs, outputs):
        """fitting data"""
        self._inputs = inputs
        self._outputs = outputs
        self.setup(inputs.shape[1], self._hidden_num, outputs.shape[1])
        self.sess.run(tf.global_variables_initializer())  # initializing all variables
        self.training()  # start training

    def training(self):
        """calculating weights through linear least square estimation and predicting"""
        weights = tf.matrix_inverse(tf.matmul(tf.transpose(self.hidden_layer), 
                                              self.hidden_layer))
        weights_1 = tf.matmul(weights, tf.transpose(self.hidden_layer))
        weights_2 = tf.matmul(weights_1, self.output_layer)
        self._weights = self.sess.run(weights_2, feed_dict={self.input_layer: self._inputs, 
                                                            self.output_layer: self._outputs})
        # predict the output, and classify output to -1 or 1 using tf.sign
        self.predictions = tf.sign(tf.matmul(self.hidden_layer, self._weights))

    def accuracy(self):
        """calculating accruacy for training dataset"""
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self._outputs), tf.float32))
        return self.sess.run(accuracy, feed_dict={self.input_layer: self._inputs, self.output_layer: self._outputs})

    def predict(self, inputs):
        """make predictions"""
        return self.sess.run(self.predictions, feed_dict={self.input_layer: inputs})

    def cal_centroid_vectors(self, inputs):
        """KMeans obtains centre vectors via unsupervised clustering based on Euclidean distance"""
        kmeans = KMeans(k=self._hidden_num, session=self.sess)
        kmeans.train(tf.constant(inputs))
        self.hidden_centers = kmeans.centers
        np.set_printoptions(suppress=True, precision=4)  # set printing format of ndarray
        print(np.array(kmeans.centers, dtype=np.float32))  # print centers obtained by kmeans clustering

    def rbfunction(self, x, c):
    	"""RBF function: exp(-||x(k)-c||^2 / (2*delta^2)), since delta is a constant, so use _delta to represent
    	(1 / (2*delta^2)) directly"""
        e_c = tf.cast(tf.expand_dims(c, 0), tf.float32)
        e_x = tf.cast(tf.expand_dims(x, 1), tf.float32)
        return tf.exp(-self._delta * tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(e_c, e_x)), 2)))
