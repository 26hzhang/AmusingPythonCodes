import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load data
mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)

# setting hyperparameters for training
learning_rate = 0.01
training_epoch = 20
batch_size = 256
display_step = 2
examples_to_show = 10

# define network parameters
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

# define input data, since it is a unsupervised learning, so only image data is needed
X = tf.placeholder(tf.float32, [None, n_input])

# define weights and biases
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}


# define autoencoder model
def encoder(x):
    # Encoder hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x):
    # Decoder hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# construct loss and optimizer
y_pred = decoder_op  # predict value
y_true = X  # actual value

# cost
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# training process
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples / batch_size)
    # start training
    train_loss = 0
    for epoch in range(training_epoch):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # run optimization op (backprop) and cost op (to get loss value)
            _, train_loss = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # print loss value each epoch
        if epoch % display_step == 0:
            print('Epoch: %04d' % epoch, 'cost=', '{:.9f}'.format(train_loss))
    print('Optimization finished')
    # use trained autoencoder on test dataset
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # compare actual image with reconstructed image
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # test dataset
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # reconstructed result
    # f.show()
    # plt.draw()
    plt.show()
    # plt.waitforbuttonpress()
