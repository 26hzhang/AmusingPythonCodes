import tensorflow as tf
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # hide messy warnings

"""
This script is for validation, which is used to compare the prediction results with RBF netowrks.

In the script, I build a multi-layer perceptron neural network to train the given training dataset and make 
predictions on test dataset.

It is a typical three layers shallow neural network: input layer -> hidden layer -> output layer. For hidden 
layer, I use 50 neurons, and for output layer, I convert the label -1 and 1 to one-hot representation, say,
-1 -> [1, 0]
 1 -> [0, 1]

From hidden layer, I use ReLU as the activation function, while the output layer the Softmax is used, since 
it is a classification task. The Stochastic Gradient Descent (SGD) is used as optmization algorithm, and the 
training loss is computed by cross entropy.
"""


def one_hot_encode(x):
    one_hot = np.array([])
    for i in range(x.shape[0]):
        if x[i][0] == -1:
            one_hot = np.append(one_hot, [1, 0])
        else:
            one_hot = np.append(one_hot, [0, 1])
    one_hot = one_hot.reshape(x.shape[0], 2)
    return one_hot


# load training dataset
train_data = scio.loadmat('./data/data_train.mat')
train_x = np.array(train_data['data_train'])
# load training label
train_label = scio.loadmat('./data/label_train.mat')
train_y = np.array(train_label['label_train'])
train_y = one_hot_encode(train_y)
# load testing dataset
test_data = scio.loadmat('./data/data_test.mat')
test_x = np.array(test_data['data_test'])

# set hyperparameters
learning_rate = 0.01  # learning rate
epochs = 500  # training epochs
batch_size = 32  # mini-batch size
# input data of shape ? x 33, label of shape ? x 1
x = tf.placeholder(tf.float32, [None, 33], name='inputs')
y = tf.placeholder(tf.float32, [None, 2], name='labels')


# weight variable initializer
def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape=shape), name=name)


# bias variable initializer
def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)


def make_prediction():
    W1 = create_weight_variable('Weight1', [33, 50])
    b1 = create_bias_variable('Bias1', [50])
    W2 = create_weight_variable('Weight2', [50, 2])
    b2 = create_bias_variable('Bias2', [2])
    hidden = tf.nn.relu(tf.matmul(x, W1) + b1)
    return tf.nn.softmax(tf.matmul(hidden, W2) + b2)


# construct model and encapsulating all ops into scopes
with tf.name_scope('predict'):
    pred = make_prediction()
with tf.name_scope('loss'):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
with tf.name_scope('optimization'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('accuracy'):
    accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))


train_loss = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initializing all variables
    # training cycle
    for epoch in range(epochs):
        avg_loss = 0.0
        total_batches = int(len(train_x) / batch_size)
        training_batch = zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size))
        for start, end in training_batch:
            batch_xs, batch_ys = train_x[start:end], train_y[start:end]
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})
            avg_loss += c / total_batches
        if (epoch + 1) % 10 == 0:
            print('Epoch %04d' % (epoch + 1), 'training cost=', '{:.9f}'.format(avg_loss))
        train_loss.append(avg_loss)
    # compute accuracy
    acc = sess.run(accuracy, feed_dict={x:train_x, y:train_y})
    print('Accuracy:', '{:.6f}'.format(acc) )
    # make predictions for test dataset
    preds = sess.run(pred, feed_dict={x:test_x})
    predicts = [-1 if preds[i][0] > preds[i][1] else 1 for i in range(len(preds))]
    print('Predictions:', predicts)

train_loss = np.array(train_loss)
index = np.arange(0, len(train_loss), 1)
plt.figure(figsize=(10, 5))
plt.plot(index, train_loss, c='r')
plt.legend(['training loss'])
plt.show()

"""
Output Sample: 500 epochs
Epoch 0010 training cost= 0.394360939
Epoch 0020 training cost= 0.308020954
Epoch 0030 training cost= 0.266316092
Epoch 0040 training cost= 0.238219076
Epoch 0050 training cost= 0.216838901
...
Epoch 0460 training cost= 0.048538214
Epoch 0470 training cost= 0.047665862
Epoch 0480 training cost= 0.046816731
Epoch 0490 training cost= 0.046003983
Epoch 0500 training cost= 0.045227259
Accuracy: 0.987879
Predictions: [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1]
"""

"""
Output Sample: 2000 epochs
Epoch 0010 training cost= 0.447845647
Epoch 0020 training cost= 0.326307867
Epoch 0030 training cost= 0.266386668
Epoch 0040 training cost= 0.230546150
Epoch 0050 training cost= 0.206672676
...
Epoch 1960 training cost= 0.012195669
Epoch 1970 training cost= 0.012116816
Epoch 1980 training cost= 0.012039535
Epoch 1990 training cost= 0.011961366
Epoch 2000 training cost= 0.011883632
Accuracy: 0.996970
Predictions: [1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1]
"""