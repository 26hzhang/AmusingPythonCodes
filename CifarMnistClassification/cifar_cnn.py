import os
import tensorflow as tf
import pickle
from .cifar_data import maybe_download_and_extract, load_training_batch


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
    pkl_path = maybe_download_and_extract()

    # load the saved dataset
    valid_features, valid_labels = pickle.load(open(os.path.join(pkl_path, 'valid.pkl'), mode='rb'))
    test_features, test_labels = pickle.load(open(os.path.join(pkl_path, 'test.pkl'), mode='rb'))

    summary_path = "./summary/summary_cifar/"
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # hyperparameters
    epochs = 100
    batch_size = 200
    learning_rate = 0.001
    init_lr = learning_rate
    lr_decay_rate = 0.05
    grad_clip = 5.0
    l2_norm_rate = 0.001

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
            conv1_filter = tf.get_variable(shape=[3, 3, 3, 32], name="conv1_filter", dtype=tf.float32)
            conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.layers.batch_normalization(conv1)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv1_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_2"):
            conv2_filter = tf.get_variable(shape=[3, 3, 32, 32], name="conv2_filter", dtype=tf.float32)
            conv2 = tf.nn.conv2d(conv1, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.layers.batch_normalization(conv2)
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.layers.dropout(conv2, rate=0.3, training=is_train)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv2_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_3"):
            conv3_filter = tf.get_variable(shape=[3, 3, 32, 64], name="conv3_filter", dtype=tf.float32)
            conv3 = tf.nn.conv2d(conv2, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.relu(conv3)
            conv3 = tf.layers.batch_normalization(conv3)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv3_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_4"):
            conv4_filter = tf.get_variable(shape=[3, 3, 64, 64], name="conv4_filter", dtype=tf.float32)
            conv4 = tf.nn.conv2d(conv3, conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv4 = tf.nn.relu(conv4)
            conv4 = tf.layers.batch_normalization(conv4)
            conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv4 = tf.layers.dropout(conv4, rate=0.3, training=is_train)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv4_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_5"):
            conv5_filter = tf.get_variable(shape=[3, 3, 64, 128], name="conv5_filter", dtype=tf.float32)
            conv5 = tf.nn.conv2d(conv4, conv5_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv5 = tf.nn.relu(conv5)
            conv5 = tf.layers.batch_normalization(conv5)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv5_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_6"):
            conv6_filter = tf.get_variable(shape=[3, 3, 128, 128], name="conv6_filter", dtype=tf.float32)
            conv6 = tf.nn.conv2d(conv5, conv6_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv6 = tf.nn.relu(conv6)
            conv6 = tf.layers.batch_normalization(conv6)
            conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv6 = tf.layers.dropout(conv6, rate=0.5, training=is_train)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv6_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_7"):
            conv7_filter = tf.get_variable(shape=[3, 3, 128, 256], name="conv7_filter", dtype=tf.float32)
            conv7 = tf.nn.conv2d(conv6, conv7_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv7 = tf.nn.relu(conv7)
            conv7 = tf.layers.batch_normalization(conv7)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv7_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("conv_layer_8"):
            conv8_filter = tf.get_variable(shape=[3, 3, 256, 256], name="conv8_filter", dtype=tf.float32)
            conv8 = tf.nn.conv2d(conv7, conv8_filter, strides=[1, 1, 1, 1], padding='SAME')
            conv8 = tf.nn.relu(conv8)
            conv8 = tf.layers.batch_normalization(conv8)
            conv8 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv8 = tf.layers.dropout(conv8, rate=0.5, training=is_train)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(conv8_filter), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("reshape_layer"):
            flat = tf.contrib.layers.flatten(conv8)

        with tf.name_scope("fc_layer_1"):
            flat_shape = flat.get_shape().as_list()
            weight = tf.get_variable(shape=[flat_shape[1], 512], name="fc1_weight", dtype=tf.float32)
            variable_summaries(weight, name="output_weight")
            bias = tf.get_variable(shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.001),
                                   name="fc1_bias")
            variable_summaries(bias, name="output_bias")
            fc1_pre = tf.matmul(flat, weight) + bias
            tf.summary.histogram('before_relu', fc1_pre)
            fc1 = tf.nn.relu(fc1_pre)
            tf.summary.histogram('after_relu', fc1)
            fc1 = tf.layers.batch_normalization(fc1)
            fc1 = tf.layers.dropout(fc1, rate=0.75, training=is_train)
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(weight), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        with tf.name_scope("fc_layer_2"):
            weight1 = tf.get_variable(shape=[512, 10], name="fc2_weight", dtype=tf.float32)
            bias1 = tf.get_variable(shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.001),
                                    name="fc2_bias")
            logits = tf.matmul(fc1, weight1) + bias1
            # l2 regularizer
            weight_loss = tf.multiply(tf.nn.l2_loss(weight1), l2_norm_rate)
            tf.add_to_collection('weight_losses', weight_loss)

        # cost
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
            weight_cost = tf.add_n(tf.get_collection('weight_losses'))
            cost += weight_cost
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
            for batch_i in range(1, 5 + 1):  # since CIFAR has 5 train batches dataset
                for batch_features, batch_labels in load_training_batch(batch_i, batch_size):
                    cur_step += 1
                    _, loss, summ = sess.run([train_op, cost, summary], feed_dict={x: batch_features, y: batch_labels,
                                                                                   is_train: True, lr: learning_rate})
                    train_writer.add_summary(summ, cur_step)
                    if cur_step % 20 == 0:
                        summ1 = sess.run(summary, feed_dict={x: valid_features, y: valid_labels, is_train: False,
                                                             lr: None})
                        valid_writer.add_summary(summ1, cur_step)
                    if cur_step % 100 == 0:
                        acc, summ2 = sess.run([accuracy, summary], feed_dict={x: test_features, y: test_labels,
                                                                              is_train: False, lr: None})
                        test_writer.add_summary(summ2, cur_step)
                        print("Step {}, test acc: {}".format(cur_step, acc))
            learning_rate = init_lr / (1 + epoch * lr_decay_rate)

        train_writer.close()
        valid_writer.close()
        test_writer.close()


if __name__ == "__main__":
    main()
