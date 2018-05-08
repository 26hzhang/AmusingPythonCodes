import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from nn_functions import multiclass_log_loss
from cnn_model import batch_norm_cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# load mnist dataset
os.makedirs("data/")
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

TASK = 'train'
checkpoint_path = 'data/MNIST_data/mnist_cnn.ckpt'  # model save path

# set hyperparameters
learning_rate = 0.01
input_size = 784
output_size = 10
batch_size = 64
steps = 5001
validate_step = 200

# Variables
inputs = tf.placeholder(tf.float32, [None, input_size])
imgs = tf.reshape(inputs, [-1, 28, 28, 1])  # reshape to 2d image
labels = tf.placeholder(tf.float32, [None, output_size])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

cost, accuracy, predict = batch_norm_cnn(imgs, labels, keep_prob, phase_train)  # build batch normalized cnn model

# Train
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
vars_to_train = tf.trainable_variables()  # option-1
vars_for_bn1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_1/batch_norm')
vars_for_bn2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_2/batch_norm')
vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))

restore_call = False
if TASK == 'test' or os.path.exists(checkpoint_path):
    restore_call = True
    vars_all = tf.all_variables()
    vars_to_init = list(set(vars_all) - set(vars_to_train))
    init = tf.variables_initializer(vars_to_init)
elif TASK == 'train':
    init = tf.global_variables_initializer()
else:
    raise ValueError('Unknown Task Switch, please check TASK switch (train | test)...')

saver = tf.train.Saver(vars_to_train)

with tf.Session() as sess:
    sess.run(init)
    if restore_call:
        saver.restore(sess, checkpoint_path)  # restore variables
    if TASK == 'train':
        print('\n Training...')
        for i in range(1, steps + 1):
            batch_imgs, batch_labels = mnist.train.next_batch(batch_size)
            train_step.run({inputs: batch_imgs, labels: batch_labels, keep_prob: 0.5, phase_train: True})
            if i % validate_step == 0:
                validate_loss, validate_accuracy = sess.run([cost, accuracy],
                                                            feed_dict={inputs: mnist.validation.images,
                                                                       labels: mnist.validation.labels,
                                                                       keep_prob: 1.0, phase_train: False})
                print('  step %5d: validation loss = %6.4f, validation accuracy = %6.4f' % (i, validate_loss,
                                                                                            validate_accuracy))

    # test model
    test_loss, test_accuracy, test_prediction = sess.run([cost, accuracy, predict],
                                                         feed_dict={inputs: mnist.test.images,
                                                                    labels: mnist.test.labels,
                                                                    keep_prob: 1.0, phase_train: False})
    print(' \nTesting...\n  test loss = %6.4f, test accuracy = %6.4f, multiclass log loss = %6.4f' %
          (test_loss, test_accuracy, multiclass_log_loss(test_prediction, mnist.test.labels)))

    # Save the variables to disk.
    if TASK == 'train':
        saver.save(sess, checkpoint_path)
