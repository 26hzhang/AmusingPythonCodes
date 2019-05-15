from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)  # load data
trX, trY, vaX, vaY, teX, teY = mnist.train.images, mnist.train.labels, mnist.validation.images, \
                               mnist.validation.labels, mnist.test.images, mnist.test.labels

summary_path = "./summary/summary_mnist_ffn/"
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

# hyperparameters
batch_size = 200
epochs = 40
learning_rate = 0.01


# define weight initializer
def init_weights(shape, stddev=0.01, name='weight'):
    return tf.Variable(tf.random_normal(shape, stddev=stddev), name=name)


# define bias initializer
def init_bias(shape, bias=0.05, name='bias'):
    return tf.Variable(tf.constant(bias, shape=shape), name=name)


with tf.Graph().as_default(), tf.Session() as sess:
    with tf.name_scope('input'):
        # set input data
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope("model"):
        weight1 = init_weights([784, 256], name="weight1")
        bias1 = init_bias([256], name="bias1")
        hidden = tf.matmul(x, weight1) + bias1
        hidden = tf.nn.relu(hidden)
        weight2 = init_weights([256, 10], name="weight2")
        bias2 = init_bias([10], name="bias2")
        y = tf.matmul(hidden, weight2) + bias2

    """Define operations"""
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    tf.summary.scalar("loss", cost)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summary_path + "train", sess.graph)
    test_writer = tf.summary.FileWriter(summary_path + 'test')

    """Training"""
    sess.run(tf.global_variables_initializer())
    test_acc = []
    cur_step = 0
    for epoch in range(epochs):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            cur_step += 1
            feed_dict = {x: trX[start:end], y_: trY[start:end]}
            _, train_loss, summ = sess.run([train_op, cost, summary], feed_dict=feed_dict)
            train_writer.add_summary(summ, cur_step)
            if cur_step % 20 == 0:
                # print("step {}, train loss: {}".format(cur_step, train_loss))
                summ1 = sess.run(summary, feed_dict={x: vaX, y_: vaY})
                test_writer.add_summary(summ1, cur_step)
            if cur_step % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: teX, y_: teY})
                test_acc.append(acc)
                print("step {}, test acc: {}".format(cur_step, acc))

print(test_acc)
x = np.arange(1, len(test_acc) + 1, 1)
plt.figure(figsize=(14, 7))  # set image size
plt.plot(x, test_acc, c='r', ls='-')
plt.xlim(0, len(test_acc) + 1)
plt.ylim(0.5, 1.0)
plt.legend(['Accuracy on Test Dataset'])
plt.grid()
plt.show()
