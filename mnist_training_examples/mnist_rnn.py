import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load data
mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)

# setting hyperparamters for training
learning_rate = 0.001
steps = 100000
batch_size = 128

# in mnist, each image has shape 28*28, so in RNN, each time step, sequence length is 28, time steps are 28
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# define input data and weight parameters
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  # shape (28, 128)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))  # shape (128, 10)
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  # shape (128, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))  # shape (10, )
}


# define a model
def rnn_model(inputs, weight, bias):
    # convert input data to (128 batch * 28 steps, 28 inputs)
    inputs = tf.reshape(inputs, [-1, n_inputs])
    # forward to hidden layer
    # x_in = (128 batch * 28 steps, 128 hidden)
    x_in = tf.matmul(inputs, weight['in']) + bias['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])
    # Using Basic LSTM Unit
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # initialize with zero, lstm is consist of two parts: (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn accepts the tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
    return tf.matmul(final_state[1], weight['out']) + bias['out']


# define loss and optimizer
pred = rnn_model(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# model predict results and compute accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < steps:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1
