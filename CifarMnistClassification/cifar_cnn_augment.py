import os
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from .cifar_data import batch_features_labels


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():

    summary_path = "./summary/summary_cifar_aug/"
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # prepare dataset
    print("load dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype(np.float32) / 255.0
    y_train = to_categorical(y_train, num_classes=10)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=12345)

    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype(np.float32) / 255.0
    y_test = to_categorical(y_test, num_classes=10)

    print("dataset augmentation...")
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format="channels_last")

    image_generator.fit(x_train, augment=False)
    x_augmented = x_train.copy()
    y_augmented = y_train.copy()
    x_augmented = image_generator.flow(x_augmented, np.zeros(x_augmented.shape[0]), batch_size=x_augmented.shape[0],
                                       shuffle=False).next()[0]

    x_train = np.concatenate((x_train, x_augmented))
    y_train = np.concatenate((y_train, y_augmented))

    x_train, y_train = utils.shuffle(x_train, y_train, random_state=0)

    # hyperparameters
    epochs = 100
    batch_size = 200
    learning_rate = 0.001
    init_lr = learning_rate
    lr_decay_rate = 0.03
    grad_clip = 5.0
    l2_norm_rate = 0.001
    res_connect = True

    with tf.Graph().as_default(), tf.Session() as sess:
        print("build model...")
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
            y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
            lr = tf.placeholder(tf.float32, name='learning_rate')
            is_train = tf.placeholder(tf.bool, shape=[], name="is_train")

        with tf.name_scope('input_reshape'):
            image_shaped_input = x
            tf.summary.image('input', image_shaped_input, 10)

        with tf.name_scope('conv_layer_1'):
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=(1, 1), padding="same", activation=tf.nn.elu,
                                     use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv1",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv1 = tf.layers.batch_normalization(conv1)

        with tf.name_scope("conv_layer_2"):
            conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv2",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv2 = conv1 + conv2 if res_connect else conv2
            conv2 = tf.layers.batch_normalization(conv2)
            conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding="same")
            conv2 = tf.layers.dropout(conv2, rate=0.2, training=is_train)

        with tf.name_scope("conv_layer_3"):
            conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv3",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope("conv_layer_4"):
            conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv4",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv4 = conv3 + conv4 if res_connect else conv4
            conv4 = tf.layers.batch_normalization(conv4)
            conv4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2, padding="same")
            conv4 = tf.layers.dropout(conv4, rate=0.3, training=is_train)

        with tf.name_scope("conv_layer_5"):
            conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv5",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv5 = tf.layers.batch_normalization(conv5)

        with tf.name_scope("conv_layer_6"):
            conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv6",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv6 = conv5 + conv6 if res_connect else conv6
            conv6 = tf.layers.batch_normalization(conv6)
            conv6 = tf.layers.max_pooling2d(conv6, pool_size=2, strides=2, padding="same")
            conv6 = tf.layers.dropout(conv6, rate=0.4, training=is_train)

        with tf.name_scope("conv_layer_7"):
            conv7 = tf.layers.conv2d(conv6, filters=256, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv7",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv7 = tf.layers.batch_normalization(conv7)

        with tf.name_scope("conv_layer_8"):
            conv8 = tf.layers.conv2d(conv7, filters=256, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,
                                     activation=tf.nn.elu, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01), name="conv8",
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            conv8 = conv7 + conv8 if res_connect else conv8
            conv8 = tf.layers.batch_normalization(conv8)
            conv8 = tf.layers.max_pooling2d(conv8, pool_size=2, strides=2, padding="same")
            conv8 = tf.layers.dropout(conv8, rate=0.5, training=is_train)

        with tf.name_scope("reshape_layer"):
            flat = tf.contrib.layers.flatten(conv8)

        with tf.name_scope("fc_layer_1"):
            flat_shape = flat.get_shape().as_list()
            weight = tf.get_variable(shape=[flat_shape[1], 512], name="fc1_weight", dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))
            variable_summaries(weight, name="output_weight")
            bias = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.001),
                                   name="fc1_bias")
            variable_summaries(bias, name="output_bias")
            fc1_pre = tf.matmul(flat, weight) + bias
            tf.summary.histogram('before_relu', fc1_pre)
            fc1 = tf.nn.relu(fc1_pre)
            tf.summary.histogram('after_relu', fc1)
            fc1 = tf.layers.batch_normalization(fc1)

        with tf.name_scope("fc_layer_2"):
            logits = tf.layers.dense(fc1, units=10, kernel_initializer=tf.glorot_uniform_initializer(), use_bias=True,
                                     bias_initializer=tf.constant_initializer(0.001), activation=None, reuse=None,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_norm_rate))

        # cost
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
            l2_loss = tf.add_n(tf.losses.get_regularization_losses())
            cost += l2_loss
        tf.summary.scalar("loss", cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        if grad_clip is not None and grad_clip > 0.0:
            grads, vs = zip(*optimizer.compute_gradients(cost))
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(cost)

        # accuracy
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32),
                                      name='accuracy')
        tf.summary.scalar("accuracy", accuracy)

        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_path + "train", sess.graph)
        valid_writer = tf.summary.FileWriter(summary_path + 'valid')
        test_writer = tf.summary.FileWriter(summary_path + "test")

        print('Training...')
        sess.run(tf.global_variables_initializer())
        # training
        cur_step = 0
        for epoch in range(epochs):
            for batch_features, batch_labels in batch_features_labels(x_train, y_train, batch_size):
                cur_step += 1
                _, loss, summ = sess.run([train_op, cost, summary], feed_dict={x: batch_features, y: batch_labels,
                                                                               is_train: True, lr: learning_rate})
                train_writer.add_summary(summ, cur_step)
                if cur_step % 20 == 0:
                    summ1 = sess.run(summary, feed_dict={x: x_valid, y: y_valid, is_train: False, lr: None})
                    valid_writer.add_summary(summ1, cur_step)
                if cur_step % 100 == 0:
                    acc, summ2 = sess.run([accuracy, summary], feed_dict={x: x_test, y: y_test, is_train: False,
                                                                          lr: None})
                    test_writer.add_summary(summ2, cur_step)
                    print("Step {}, test acc: {}".format(cur_step, acc))
            if epoch >= 25:
                learning_rate = init_lr / (1 + epoch * lr_decay_rate)

        train_writer.close()
        valid_writer.close()
        test_writer.close()


if __name__ == "__main__":
    main()
