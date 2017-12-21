from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)  # load data

# create model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b  # prediction
y_ = tf.placeholder(tf.float32, [None, 10])  # label

# define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# using InteractiveSession to create interactive contextual session different from normal session, interactive session
# will become default session functions, like tf.Tensor.eval and tf.Operation.run, can be operated by this session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())  # initialize all the Variables

for _ in range(1000):  # training
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# test trained model
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
