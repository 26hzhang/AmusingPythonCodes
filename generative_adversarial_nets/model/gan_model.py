import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt


class GAN(object):
    def __init__(self, input_size, gen_hidden_size, disc_hidden_size, noise_size, learning_rate=2e-4,
                 model_name='gan_model', ckpt_path='./data/ckpt/gan/'):
        self.input_size = input_size
        self.gen_hidden_size = gen_hidden_size
        self.disc_hidden_size = disc_hidden_size
        self.noise_size = noise_size
        self.learning_rate = learning_rate
        self.model_name = model_name  # model name
        self.ckpt_path = ckpt_path  # checkpoint path, to save model
        # build graph
        sys.stdout.write('\nBuilding Graph...')
        tf.reset_default_graph()
        # set inputs
        self.gen_input = tf.placeholder(tf.float32, shape=[None, self.noise_size], name='input_noise')
        self.disc_input = tf.placeholder(tf.float32, shape=[None, self.input_size], name='disc_input')

        def __init_weight__(shape):  # a custom initialization (see xavier glorot init)
            return tf.Variable(tf.random_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2.0)))

        def __init_bias__(shape):  # initialize bias with zeros
            return tf.Variable(tf.zeros(shape=shape))

        # store layers weight & bias
        self.weights = {'gen_hidden1': __init_weight__([self.noise_size, self.gen_hidden_size]),
                        'gen_out': __init_weight__([self.gen_hidden_size, self.input_size]),
                        'disc_hidden1': __init_weight__([self.input_size, self.disc_hidden_size]),
                        'disc_out': __init_weight__([self.disc_hidden_size, 1])}
        self.biases = {'gen_hidden1': __init_bias__([self.gen_hidden_size]),
                       'gen_out': __init_bias__([self.input_size]),
                       'disc_hidden1': __init_bias__([self.disc_hidden_size]),
                       'disc_out': __init_bias__([1])}
        # build Generator network
        self.gen_sample = self.__generator__(self.gen_input)
        # build 2 Discriminator networks (one from noise input, one from generated samples)
        disc_real = self.__discriminator__(self.disc_input)
        disc_fake = self.__discriminator__(self.gen_sample)
        # build Loss
        self.gen_loss = -tf.reduce_mean(tf.log(disc_fake))
        self.disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1.0 - disc_fake))
        # training Variables for each optimizer, by default in TensorFlow, all variables are updated by each optimizer,
        # so we need to precise for each one of them the specific variables to update.
        gen_vars = [self.weights['gen_hidden1'], self.weights['gen_out'], self.biases['gen_hidden1'],
                    self.biases['gen_out']]  # Generator network variables
        disc_vars = [self.weights['disc_hidden1'], self.weights['disc_out'], self.biases['disc_hidden1'],
                     self.biases['disc_out']]  # Discriminator network variables
        # build optimizers and create training operations
        self.train_gen = tf.train.AdamOptimizer(self.learning_rate).minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = tf.train.AdamOptimizer(self.learning_rate).minimize(self.disc_loss, var_list=disc_vars)
        sys.stdout.write(' Done...\n')

    def __generator__(self, x):  # generator part
        hidden_layer = tf.add(tf.matmul(x, self.weights['gen_hidden1']), self.biases['gen_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.add(tf.matmul(hidden_layer, self.weights['gen_out']), self.biases['gen_out'])
        return tf.nn.sigmoid(out_layer)

    def __discriminator__(self, x):  # discriminator part
        hidden_layer = tf.add(tf.matmul(x, self.weights['disc_hidden1']), self.biases['disc_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.add(tf.matmul(hidden_layer, self.weights['disc_out']), self.biases['disc_out'])
        return tf.nn.sigmoid(out_layer)

    def train(self, dataset, num_steps=100000, batch_size=128, sess=None):
        saver = tf.train.Saver(max_to_keep=1)  # save the model periodically
        if not sess:  # if no session is given
            sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # init all variables
        sys.stdout.write('Training started...\n')
        i = 0
        try:
            for i in range(1, num_steps + 1):
                # Get the next batch of MNIST data (only images are needed, not labels)
                batch_x, _ = dataset.train.next_batch(batch_size)
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, self.noise_size])
                feed_dict = {self.disc_input: batch_x, self.gen_input: z}
                _, _, gl, dl = sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                        feed_dict=feed_dict)
                if i % 2000 == 0 or i == 1:
                    print('Step %6d: Generator Loss: %8.7f, Discriminator Loss: %8.7f' % (i, gl, dl))
                if i % 20000 == 0:  # save the model periodically
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
        # generate images from noise, using the generator network.
        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            # noise input.
            z = np.random.uniform(-1., 1., size=[4, self.noise_size])
            g = sess.run([self.gen_sample], feed_dict={self.gen_input: z})
            g = np.reshape(g, newshape=(4, 28, 28, 1))
            # reverse colours for better display
            g = -1 * (g - 1)
            for j in range(4):
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
                a[j][i].imshow(img)
        f.show()

    def generate2(self, sess):
        n = 6
        canvas = np.empty((28 * n, 28 * n))
        for i in range(n):
            # noise input.
            z = np.random.uniform(-1., 1., size=[n, self.noise_size])
            # generate image from noise.
            g = sess.run(self.gen_sample, feed_dict={self.gen_input: z})
            # reverse colours for better display
            g = -1 * (g - 1)
            for j in range(n):
                # draw the generated digits
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])
        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()
