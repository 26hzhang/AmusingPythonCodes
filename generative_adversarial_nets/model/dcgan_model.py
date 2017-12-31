import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys


class DCGAN(object):
    def __init__(self, image_size, noise_size, learning_rate=1e-3, model_name='dcgan_model',
                 ckpt_path='./data/ckpt/dcgan/'):
        self.image_size = image_size
        self.noise_size = noise_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        # build graph
        sys.stdout.write('\nBuilding Graph...')
        tf.reset_default_graph()
        # set inputs
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_size])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        # build targets (real or fake images)
        self.disc_target = tf.placeholder(tf.int32, shape=[None])
        self.gen_target = tf.placeholder(tf.int32, shape=[None])
        # build generator network
        self.gen_sample = self.__generator__(self.noise_input)
        # build 2 discriminator networks (one from noise input, one from generated samples)
        disc_real = self.__discriminator__(self.real_image_input)
        disc_fake = self.__discriminator__(self.gen_sample, reuse=True)
        disc_concat = tf.concat([disc_real, disc_fake], axis=0)
        # build the stacked generator/discriminator
        stacked_gan = self.__discriminator__(self.gen_sample, reuse=True)
        # build loss
        self.disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_concat,
                                                                                       labels=self.disc_target))
        self.gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=stacked_gan,
                                                                                      labels=self.gen_target))
        # build Optimizers
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # training variables for each optimizer, by default in TensorFlow, all variables are updated by each optimizer,
        # so we need to precise for each one of them the specific variables to update.
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')  # generator vars
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')  # discriminator vars
        # create training operations
        self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)
        sys.stdout.write(' Done...\n')

    @staticmethod
    def __generator__(x, reuse=False):  # generator network, input: noise, output: image
        with tf.variable_scope('Generator', reuse=reuse):
            # TensorFlow Layers automatically create variables and calculate their shape, based on the input.
            x = tf.layers.dense(x, units=6 * 6 * 128)
            x = tf.nn.tanh(x)
            # reshape to a 4-D array of images: (batch, height, width, channels): (batch, 6, 6, 128)
            x = tf.reshape(x, shape=[-1, 6, 6, 128])
            x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)  # deconvolution, image shape: (batch, 14, 14, 64)
            x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)  # deconvolution, image shape: (batch, 28, 28, 1)
            x = tf.nn.sigmoid(x)  # apply sigmoid to clip values between 0 and 1
        return x

    @staticmethod
    def __discriminator__(x, reuse=False):  # discriminator network, input: image, output: prediction real/fake image
        with tf.variable_scope('Discriminator', reuse=reuse):  # typical cnn to classify images.
            x = tf.layers.conv2d(x, 64, 5)
            x = tf.nn.tanh(x)
            x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.layers.conv2d(x, 128, 5)
            x = tf.nn.tanh(x)
            x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.nn.tanh(x)
            x = tf.layers.dense(x, 2)  # output 2 classes: real and fake images
        return x

    def train(self, trainset, num_steps=20000, batch_size=64, sess=None):
        saver = tf.train.Saver(max_to_keep=1)  # save the model periodically
        if not sess:  # if no session is given
            sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # init all variables
        sys.stdout.write('Training started...\n')
        i = 0
        try:
            for i in range(1, num_steps + 1):
                batch_x, _ = trainset.train.next_batch(batch_size)
                batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
                # generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, self.noise_size])
                # prepare targets (real image: 1, fake image: 0)
                # the first half of data fed to the generator are real images,
                # the other half are fake images (coming from the generator).
                batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
                # generator tries to fool the discriminator, thus targets are 1
                batch_gen_y = np.ones([batch_size])
                # training
                feed_dict = {self.real_image_input: batch_x, self.noise_input: z, self.disc_target: batch_disc_y,
                             self.gen_target: batch_gen_y}
                _, _, gl, dl = sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                        feed_dict=feed_dict)
                if i % 200 == 0 or i == 1:
                    print('Step %6d: Generator Loss: %8.7f, Discriminator Loss: %8.7f' % (i, gl, dl))
                if i % 2000 == 0:  # save the model periodically
                    sys.stdout.write('Saving model at step %d... ' % i)
                    saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
                    sys.stdout.write('Done...\n')
        except KeyboardInterrupt:
            sys.stdout.write('Interrupted by user at training step %d, saving model at this step.. ' % i)
        saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
        sys.stdout.write('Done...\n')
        return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()  # create a session
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess

    def generate(self, sess):
        # generate images from noise, using the generator network
        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            # noise input
            z = np.random.uniform(-1., 1., size=[4, self.noise_size])
            g = sess.run(self.gen_sample, feed_dict={self.noise_input: z})
            for j in range(4):
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
                a[j][i].imshow(img)
        f.show()

    def generate2(self, sess):
        n = 6
        canvas = np.empty((28 * n, 28 * n))
        for i in range(n):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[n, self.noise_size])
            # Generate image from noise.
            g = sess.run(self.gen_sample, feed_dict={self.noise_input: z})
            # Rescale values to the original [0, 1] (from tanh -> [-1, 1])
            g = (g + 1.) / 2.
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(n):
                # Draw the generated digits
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])
        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()


class DCGANBatchNorm(object):
    def __init__(self, image_dim, noise_dim, batch_size=128, learning_rate_gen=0.002, learning_rate_disc=0.002,
                 model_name='dcgan_batch_model', ckpt_path='./data/ckpt/dcganbatch/'):
        self.image_dim = image_dim
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.learning_rate_generator = learning_rate_gen
        self.learning_rate_discriminator = learning_rate_disc
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        # build graph
        sys.stdout.write('\nBuilding Graph...')
        tf.reset_default_graph()
        # set inputs
        self.noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        # a boolean to indicate batch normalization if it is training or inference time
        self.is_training = tf.placeholder(tf.bool)

        # LeakyReLU activation
        def __leakyrelu__(x, alpha=0.2):
            return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

        # Generator Network, Input: Noise, Output: Image
        # Note that batch normalization has different behavior at training and inference time,
        # we then use a placeholder to indicates the layer if we are training or not.
        def __generator__(x, reuse=False):
            with tf.variable_scope('Generator', reuse=reuse):
                # TensorFlow Layers automatically create variables and calculate their
                # shape, based on the input.
                x = tf.layers.dense(x, units=7 * 7 * 128)
                x = tf.layers.batch_normalization(x, training=self.is_training)
                x = tf.nn.relu(x)
                # Reshape to a 4-D array of images: (batch, height, width, channels)
                # New shape: (batch, 7, 7, 128)
                x = tf.reshape(x, shape=[-1, 7, 7, 128])
                # Deconvolution, image shape: (batch, 14, 14, 64)
                x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=self.is_training)
                x = tf.nn.relu(x)
                # Deconvolution, image shape: (batch, 28, 28, 1)
                x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')
                # Apply tanh for better stability - clip values to [-1, 1].
                x = tf.nn.tanh(x)
                return x

        # Discriminator Network, Input: Image, Output: Prediction Real/Fake Image
        def __discriminator__(x, reuse=False):
            with tf.variable_scope('Discriminator', reuse=reuse):
                # Typical convolutional neural network to classify images.
                x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=self.is_training)
                x = __leakyrelu__(x)
                x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=self.is_training)
                x = __leakyrelu__(x)
                # Flatten
                x = tf.reshape(x, shape=[-1, 7 * 7 * 128])
                x = tf.layers.dense(x, 1024)
                x = tf.layers.batch_normalization(x, training=self.is_training)
                x = __leakyrelu__(x)
                # Output 2 classes: Real and Fake images
                x = tf.layers.dense(x, 2)
            return x

        # Build Generator Network
        self.gen_sample = __generator__(self.noise_input)
        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = __discriminator__(self.real_image_input)
        disc_fake = __discriminator__(self.gen_sample, reuse=True)
        # Build the stacked generator/discriminator
        stacked_gan = __discriminator__(self.gen_sample, reuse=True)
        # Build Loss (Labels for real images: 1, for fake images: 0)
        # Discriminator Loss for real and fake samples
        disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones([self.batch_size], dtype=tf.int32)))
        disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros([self.batch_size], dtype=tf.int32)))
        # Sum both loss
        self.disc_loss = disc_loss_real + disc_loss_fake
        # Generator Loss (The generator tries to fool the discriminator, thus labels are 1)
        self.gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_gan, labels=tf.ones([self.batch_size], dtype=tf.int32)))
        # Build Optimizers
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate_generator, beta1=0.5, beta2=0.999)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate_discriminator, beta1=0.5, beta2=0.999)
        # Training Variables for each optimizer, By default in TensorFlow, all variables are updated by each optimizer,
        # so we need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator Network Variables
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        # Create training operations
        # TensorFlow UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev
        gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
        # `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)
        with tf.control_dependencies(gen_update_ops):
            self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
        with tf.control_dependencies(disc_update_ops):
            self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)
        sys.stdout.write(' Done...\n')

    def train(self, dataset, num_steps=10000, sess=None):
        saver = tf.train.Saver(max_to_keep=1)  # save the model periodically
        if not sess:  # if no session is given
            sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # init all variables
        sys.stdout.write('Training started...\n')
        i = 0
        try:
            for i in range(1, num_steps + 1):
                # Prepare Input Data
                # Get the next batch of MNIST data (only images are needed, not labels)
                batch_x, _ = dataset.train.next_batch(self.batch_size)
                batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
                # Rescale to [-1, 1], the input range of the discriminator
                batch_x = batch_x * 2.0 - 1.0
                # Discriminator Training
                # Generate noise to feed to the generator
                z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
                _, dl = sess.run([self.train_disc, self.disc_loss], feed_dict={self.real_image_input: batch_x,
                                                                               self.noise_input: z,
                                                                               self.is_training: True})
                # Generator Training
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[self.batch_size, self.noise_dim])
                _, gl = sess.run([self.train_gen, self.gen_loss], feed_dict={self.noise_input: z,
                                                                             self.is_training: True})
                if i % 500 == 0 or i == 1:
                    print('Step %6d: Generator Loss: %8.7f, Discriminator Loss: %8.7f' % (i, gl, dl))
                if i % 2000 == 0:  # save the model periodically
                    sys.stdout.write('Saving model at step %d... ' % i)
                    saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
                    sys.stdout.write('Done...\n')
        except KeyboardInterrupt:
            sys.stdout.write('Interrupted by user at training step %d, saving model at this step.. ' % i)
        saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
        sys.stdout.write('Done...\n')
        return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()  # create a session
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess

    def generate(self, sess):
        n = 6
        canvas = np.empty((28 * n, 28 * n))
        for i in range(n):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[n, self.noise_dim])
            # Generate image from noise.
            g = sess.run(self.gen_sample, feed_dict={self.noise_input: z, self.is_training: False})
            # Rescale values to the original [0, 1] (from tanh -> [-1, 1])
            g = (g + 1.) / 2.
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(n):
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])
        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()
