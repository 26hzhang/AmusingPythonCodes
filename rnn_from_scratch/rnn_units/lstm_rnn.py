import tensorflow as tf
import numpy as np
import sys
import random


class LstmRNN(object):
    def __init__(self, num_classes, state_size, learning_rate=0.1, model_name='lstm_rnn_model',
                 ckpt_path='./ckpt/lstm/'):
        self.num_classes = num_classes
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        # build graph
        sys.stdout.write('\nBuilding Graph...')
        tf.reset_default_graph()
        # set inputs
        self.xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
        # embeddings
        embs = tf.get_variable('emb', [self.num_classes, self.state_size])
        rnn_inputs = tf.nn.embedding_lookup(embs, self.xs_)
        # initial hidden state
        self.init_state = tf.placeholder(shape=[2, None, self.state_size], dtype=tf.float32, name='initial_state')
        # initializer and params
        xav_init = tf.contrib.layers.xavier_initializer
        w = tf.get_variable('W', shape=[4, self.state_size, self.state_size], initializer=xav_init())
        u = tf.get_variable('U', shape=[4, self.state_size, self.state_size], initializer=xav_init())

        def __step__(prev, x):
            # gather previous internal state and output state
            st_1, ct_1 = tf.unstack(prev)
            #  input gate
            i = tf.sigmoid(tf.matmul(x, u[0]) + tf.matmul(st_1, w[0]))
            #  forget gate
            f = tf.sigmoid(tf.matmul(x, u[1]) + tf.matmul(st_1, w[1]))
            #  output gate
            o = tf.sigmoid(tf.matmul(x, u[2]) + tf.matmul(st_1, w[2]))
            #  gate weights
            g = tf.tanh(tf.matmul(x, u[3]) + tf.matmul(st_1, w[3]))
            # new internal cell state
            ct = ct_1 * f + g * i
            # output state
            st = tf.tanh(ct) * o
            return tf.stack([st, ct])

        states = tf.scan(__step__, tf.transpose(rnn_inputs, [1, 0, 2]), initializer=self.init_state)
        # predictions
        v = tf.get_variable('v', shape=[self.state_size, self.num_classes], initializer=xav_init())
        bo = tf.get_variable('bo', shape=[self.num_classes], initializer=tf.constant_initializer(0.0))
        # get last state before reshape/transpose
        self.last_state = states[-1]
        # transpose and reshape
        states = tf.reshape(tf.transpose(states, [1, 2, 0, 3])[0], [-1, self.state_size])
        logits = tf.add(tf.matmul(states, v), bo)
        # predictions
        self.predictions = tf.nn.softmax(logits)
        # optimization
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ys_))
        self.train_op = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        sys.stdout.write(' Done...\n')

    def train(self, train_set, epochs=50, steps_per_epoch=100):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            epoch = 0
            try:
                for epoch in range(epochs):
                    for step in range(steps_per_epoch):
                        xs, ys = train_set.__next__()
                        batch_size = xs.shape[0]
                        feed_dict = {self.xs_: xs, self.ys_: ys.flatten(),
                                     self.init_state: np.zeros([2, batch_size, self.state_size])}
                        _, cur_loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    print('Epoch [{}], average loss : {}'.format(epoch, train_loss / steps_per_epoch))
                    train_loss = 0
            except KeyboardInterrupt:
                print('Interrupted by user at ' + str(epoch))
            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + self.model_name, global_step=epoch)

    def generate(self, idx2w, w2idx, num_words=100, separator=' '):
        # generate text
        random_init_word = random.choice(idx2w)
        current_word = w2idx[random_init_word]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            words = [current_word]
            state = None
            state_ = None
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = {self.xs_: np.array([current_word]).reshape([1, 1]), self.init_state: state_}
                else:
                    feed_dict = {self.xs_: np.array([current_word]).reshape([1, 1]),
                                 self.init_state: np.zeros([2, 1, self.state_size])}
                # forward propagation
                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
                # set flag to true
                state = True
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)
        return separator.join([idx2w[w] for w in words])
