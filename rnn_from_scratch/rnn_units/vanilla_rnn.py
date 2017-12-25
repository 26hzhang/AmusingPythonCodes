import tensorflow as tf
import numpy as np
import sys
import random


class VanillaRNN(object):
    def __init__(self, num_classes, state_size, seq_len, learning_rate=0.1, model_name='vanilla_rnn_model',
                 ckpt_path='./ckpt/vanilla/'):
        self.num_classes = num_classes
        self.state_size = state_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        # build graph
        sys.stdout.write('\nBuilding Graph...')
        tf.reset_default_graph()
        # set feed data
        self.xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
        # embeddings
        embs = tf.get_variable('emb', [self.num_classes, self.state_size])
        rnn_inputs = tf.nn.embedding_lookup(embs, self.xs_)
        # initial hidden state
        self.init_state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='initial_state')
        # initializer
        xav_init = tf.contrib.layers.xavier_initializer

        # define step operation: St = tanh(U * Xt + W * St-1)
        def __step__(h_prev, x_curr):
            w = tf.get_variable('W', shape=[self.state_size, self.state_size], initializer=xav_init())
            u = tf.get_variable('U', shape=[self.state_size, self.state_size], initializer=xav_init())
            b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.0))
            h = tf.tanh(tf.matmul(h_prev, w) + tf.matmul(x_curr, u) + b)
            return h

        # here comes the scan operation; tf.scan(fn, elems, initializer)
        # repeatedly applies the callable function (__step__) to a sequence of elements from first to last
        # ref: https://www.tensorflow.org/api_docs/python/tf/scan
        states = tf.scan(__step__, tf.transpose(rnn_inputs, [1, 0, 2]), initializer=self.init_state)
        # set last state
        self.last_state = states[-1]

        # predictions
        v = tf.get_variable('V', shape=[self.state_size, self.num_classes], initializer=xav_init())
        bo = tf.get_variable('bo', shape=[self.num_classes], initializer=tf.constant_initializer(0.0))
        # transpose and flatten states to 2d matrix for matmult with V
        states = tf.reshape(tf.transpose(states, [1, 0, 2]), [-1, self.state_size])
        logits = tf.add(tf.matmul(states, v), bo)
        self.predictions = tf.nn.softmax(logits)  # Yt = softmax(V * St)
        # optimization
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ys_))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        sys.stdout.write(' Done...\n')

    # training operation
    def train(self, train_set, epochs=50, steps_per_epoch=500):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            epoch = 0
            try:
                for epoch in range(epochs):
                    for step in range(steps_per_epoch):
                        xs, ys = train_set.__next__()
                        batch_size = xs.shape[0]
                        feed_dict = {self.xs_: xs, self.ys_: ys.reshape([batch_size * self.seq_len]),
                                     self.init_state: np.zeros([batch_size, self.state_size])}
                        _, cur_loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                        train_loss += cur_loss
                    print('Epoch [{}], average loss : {}'.format(epoch, train_loss / steps_per_epoch))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at epoch ' + str(epoch))
            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + 'vanilla.ckpt', global_step=epoch)

    # generating operation
    def generate(self, idx2ch, ch2idx, num_words=100, separator=' '):
        # generate text
        random_init_word = random.choice(idx2ch)
        current_word = ch2idx[random_init_word]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # generate operation
            words = [current_word]
            state = None
            state_ = None
            # set batch_size to 1
            batch_size = 1
            num_words = num_words if num_words else 111
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = {self.xs_: np.array(current_word).reshape([1, 1]), self.init_state: state_}
                else:
                    feed_dict = {self.xs_: np.array(current_word).reshape([1, 1]),
                                 self.init_state: np.zeros([batch_size, self.state_size])}
                # forward propagation
                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
                # set flag to true
                state = True
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)
        return separator.join([idx2ch[w] for w in words])
