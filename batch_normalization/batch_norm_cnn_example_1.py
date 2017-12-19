import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from nn_functions import multiclass_log_loss
from cnn_model import batch_norm_cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# define flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'train', '[train | test], default train')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

# load mnist dataset
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', './data/models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', './data/checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', './data/logs/')

# set hyperparameters
learning_rate = 0.01
input_size = 784
output_size = 10
batch_size = 64
steps = 5001
validate_interval = 200

with tf.Graph().as_default(), tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [None, input_size])
    imgs = tf.reshape(inputs, [-1, 28, 28, 1])  # reshape to 2d image
    labels = tf.placeholder(tf.float32, [None, output_size])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # build model
    cost, accuracy, predict = batch_norm_cnn(imgs, labels, keep_prob, phase_train)

    # add summaries
    merge_summaries = tf.summary.merge_all()

    # create trainer
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=5)

    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph, model_path, 'bn_cnn.pb', as_text=True)

    # restore, if possible
    if FLAGS.restore:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(sess, latest_checkpoint_path)

    # training
    if FLAGS.task == 'train':
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        print('\n Training...')
        for step in range(steps):
            batch_imgs, batch_labels = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train_step], feed_dict={inputs: batch_imgs, labels: batch_labels, keep_prob: 0.5,
                                                           phase_train: True})
            if step % validate_interval == 0:
                validate_loss, validate_accuracy, summary = sess.run([cost, accuracy, merge_summaries],
                                                                     feed_dict={inputs: mnist.validation.images,
                                                                                labels: mnist.validation.labels,
                                                                                keep_prob: 1.0, phase_train: False})
                summary_writer.add_summary(summary, step)
                saver.save(sess, checkpoint_path + 'checkpoint', global_step=step)
                print('  step %5d: validation loss = %6.4f, validation accuracy = %6.4f' % (step, validate_loss,
                                                                                            validate_accuracy))

    # test model
    print('\n Testing...')
    test_loss, test_accuracy, test_prediction = sess.run([cost, accuracy, predict],
                                                         feed_dict={inputs: mnist.test.images,
                                                                    labels: mnist.test.labels,
                                                                    keep_prob: 1.0, phase_train: False})
    print('  test loss = %6.4f, test accuracy = %6.4f, multiclass log loss = %6.4f' %
          (test_loss, test_accuracy, multiclass_log_loss(test_prediction, mnist.test.labels)))
