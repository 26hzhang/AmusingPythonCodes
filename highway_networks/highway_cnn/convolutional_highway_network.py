import tensorflow as tf
import numpy as np
import os
from highway_networks.highway_full_connect.layers import dense_layer
from highway_networks.highway_cnn.layers import conv2d_layer, conv2d_highway_layer
from tensorflow.examples.tutorials.mnist import input_data


# define flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('skip_training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', '../data/models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', '../data/checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', '../data/logs/')

# read data
mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)
# hyperparameters
learning_rate = 0.01  # learning rate
batch_size = 64  # mini-batch size for each train step
checkpoint_interval = 100  # steps to validate model and save model
epochs = 5  # training epochs
input_shape = 784  # dim of input data
output_shape = 10  # dim of output target

# create graph
with tf.Graph().as_default(), tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [None, input_shape])
    targets = tf.placeholder(tf.float32, [None, output_shape])
    images = tf.reshape(inputs, [-1, 28, 28, 1])  # reshape image to 2-D
    """A 18-layers highway convolutional neural networks"""
    # first
    keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')
    x_drop = tf.nn.dropout(images, keep_prob=keep_prob1)
    prev_layer = conv2d_layer(x_drop, [5, 5, 1, 32], [32], [1, 1, 1, 1], 'SAME')
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    # second
    keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')
    prev_layer = tf.nn.dropout(prev_layer, keep_prob=keep_prob2)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # third
    keep_prob3 = tf.placeholder(tf.float32, name='keep_prob3')
    prev_layer = tf.nn.dropout(prev_layer, keep_prob=keep_prob3)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = conv2d_highway_layer(prev_layer, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=-1.0)
    prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # fourth
    keep_prob4 = tf.placeholder(tf.float32, name='keep_prob4')
    prev_layer = tf.nn.dropout(prev_layer, keep_prob=keep_prob4)
    prev_layer = tf.reshape(prev_layer, [-1, 4 * 4 * 32])
    outputs = dense_layer(prev_layer, 4 * 4 * 32, 10, tf.nn.softmax)

    """Set training operations"""
    # define training and accuracy operations
    with tf.name_scope('loss'):
        cost = -tf.reduce_sum(targets * tf.log(outputs))
        tf.summary.scalar('loss', cost)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merge_summaries = tf.summary.merge_all()

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    """Training and testing"""
    sess.run(tf.global_variables_initializer())
    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, model_path, 'highway.pb', as_text=True)
    # restore, if possible
    if FLAGS.restore:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(sess, latest_checkpoint_path)

    # training
    if not FLAGS.skip_training:
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph_def)
        step = 0
        for epoch in range(epochs):
            train_cost = []
            for batch_index in range(mnist.train.num_examples // batch_size):
                step += 1
                batch_imgs, batch_labels = mnist.train.next_batch(batch_size)
                if step % checkpoint_interval == 0:
                    validate_accuracy, summary = sess.run([accuracy, merge_summaries], feed_dict={
                        inputs: mnist.validation.images,
                        targets: mnist.validation.labels,
                        keep_prob1: 1.0, keep_prob2: 1.0,
                        keep_prob3: 1.0, keep_prob4: 1.0})
                    summary_writer.add_summary(summary, step)
                    saver.save(sess, checkpoint_path + 'checkpoint', global_step=step)
                    print('step %d, validating accuracy %g' % (step, validate_accuracy))
                c, _ = sess.run([cost, optimizer], feed_dict={inputs: batch_imgs, targets: batch_labels, keep_prob1: 0.8,
                                                              keep_prob2: 0.7, keep_prob3: 0.6, keep_prob4: 0.5})
                train_cost.append(c)
            print("Epoch: {}/{}".format(epoch + 1, epochs), "  |  Current loss: {:9.6f}".format(np.mean(train_cost)))
            print("Test accuracy %g" % sess.run(accuracy, feed_dict={inputs: mnist.test.images,
                                                                     targets: mnist.test.labels,
                                                                     keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,
                                                                     keep_prob4: 1.0}))
        summary_writer.close()

    # testing
    test_accuracy = sess.run(accuracy, feed_dict={inputs: mnist.test.images, targets: mnist.test.labels,
                                                  keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, keep_prob4: 1.0})
    print('Test accuracy %g' % test_accuracy)
