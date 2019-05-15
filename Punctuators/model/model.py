import tensorflow as tf
import os
import codecs
import numpy as np
from numpy import nan
from tqdm import tqdm
from utils.data_utils import batch_iter, pad_sequences
from tensorflow.python.ops.rnn import dynamic_rnn
from model.nns import viterbi_decode, dense, highway_network
from model.rnns import BiRNN, DenseConnectBiRNN, AttentionCell
from utils.error_calculator import compute_error
from utils.logger import get_logger, Progbar
from data.data import PUNCTUATION_MAPPING, END, UNK, EOS_TOKENS, SPACE


class Punctuator(object):
    def __init__(self, config):
        self.cfg = config
        self.logger = get_logger(os.path.join(self.cfg.ckpt_path, 'log.txt'))
        self.sess, self.saver, self.start_epoch = None, None, 1
        self._add_placeholder()
        self._build_model_op()
        self._add_loss_op()
        self._build_train_op()
        print('params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        self.initialize_session()

    def reinitialize_weights(self, scope_name):
        """Reinitialize parameters in a scope"""
        variables = tf.contrib.framework.get_variables(scope_name)
        self.sess.run(tf.variables_initializer(variables))

    def initialize_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(tf.global_variables_initializer())
        if self.cfg.resume_training:
            ch = tf.train.get_checkpoint_state(self.cfg.ckpt_path)
            if not ch:
                print('No checkpoint found in directory %s. Starting a new training session...' % self.cfg.ckpt_path)
                return
            ckpt_path = ch.model_checkpoint_path
            self.start_epoch = int(ckpt_path.split('-')[-1]) + 1
            print("Resuming training from {}, start epoch: {}".format(self.cfg.model_save_path, self.start_epoch))
            self.saver.restore(self.sess, ckpt_path)

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        if not os.path.exists(self.cfg.ckpt_path):
            os.makedirs(self.cfg.ckpt_path)
        self.saver.save(self.sess, self.cfg.model_save_path, global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_placeholder(self):
        # shape = (batch_size, max_time)
        self.x = tf.placeholder(tf.int32, shape=[None, None], name='x')
        # shape = (batch_size)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        # shape = (batch_size, max_time, max_word_length)
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name='chars')
        # shape = (batch_size, max_time)
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name='char_seq_len')
        # shape = (batch_size, max_time - 1)
        self.y = tf.placeholder(tf.int32, shape=[None, None], name='y')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

    def _get_feed_dict(self, x, seq_len, chars, char_seq_len, y=None, is_train=None):
        feed_dict = {self.x: x, self.seq_len: seq_len, self.chars: chars, self.char_seq_len: char_seq_len}
        if y is not None:
            feed_dict[self.y] = y
        if is_train is not None:
            feed_dict[self.is_train] = is_train
        return feed_dict

    def _build_model_op(self):
        with tf.variable_scope('embeddings'):
            if self.cfg.embedding is not None:
                # pretrained GloVe embeddings
                _word_emb_1 = tf.Variable(self.cfg.embedding, dtype=tf.float32, name='_word_emb_1', trainable=False)
                # special tokens (UNK, NUM and END) for training
                _word_emb_2 = tf.get_variable(name='_word_emb_2', shape=[3, self.cfg.emb_dim], dtype=tf.float32,
                                              trainable=True)
                _word_emb = tf.concat([_word_emb_1, _word_emb_2], axis=0, name='_word_emb')
            else:
                _word_emb = tf.get_variable(name='_word_emb', shape=[self.cfg.vocab_size, self.cfg.emb_dim],
                                            dtype=tf.float32, trainable=True)
            word_emb = tf.nn.embedding_lookup(_word_emb, self.x, name='word_emb')
            if self.cfg.char_emb is not None:
                _char_emb = tf.Variable(self.cfg.char_emb, dtype=tf.float32, name='_char_emb', trainable=True)
            else:
                _char_emb = tf.get_variable(name='_char_emb', shape=[self.cfg.char_vocab_size, self.cfg.char_emb_dim],
                                            dtype=tf.float32, trainable=True)
            char_emb = tf.nn.embedding_lookup(_char_emb, self.chars, name="char_emb")
            with tf.variable_scope('char_represent'):
                char_rnn = BiRNN(self.cfg.char_num_units)
                char_out = char_rnn(char_emb, self.char_seq_len, return_last_state=True)
            emb = tf.concat([word_emb, char_out], axis=-1)
            print('word_emb shape: {}'.format(emb.get_shape().as_list()))

        with tf.variable_scope('highway'):
            emb = highway_network(emb, 2, bias=True, keep_prob=self.cfg.keep_prob, is_train=self.is_train)

        with tf.variable_scope('encoder'):
            encoder = DenseConnectBiRNN(self.cfg.num_layers, self.cfg.num_units_list)
            context = encoder(emb, seq_len=self.seq_len)
            print('context shape: {}'.format(context.get_shape().as_list()))

        with tf.variable_scope('attention'):
            proj_context = dense(context, 2 * self.cfg.num_units)
            context = tf.transpose(context, [1, 0, 2])
            proj_context = tf.transpose(proj_context, [1, 0, 2])
            att_cell = AttentionCell(self.cfg.num_units, context, proj_context)
            att, _ = dynamic_rnn(att_cell, context[1:, :, :], sequence_length=self.seq_len - 1, dtype=tf.float32,
                                 time_major=True)
            att = tf.transpose(att, [1, 0, 2])
            print('attention shape: {}'.format(att.get_shape().as_list()))

        with tf.variable_scope('project'):
            self.logits = dense(att, self.cfg.label_size, use_bias=True, scope='dense')
            print("logits shape: {}".format(self.logits.get_shape().as_list()))

    def _add_loss_op(self):
        log_ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.y, self.seq_len - 1)
        if self.cfg.l2_reg is not None and self.cfg.l2_reg > 0.0:  # l2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            self.loss = tf.reduce_mean(-log_ll) + self.cfg.l2_reg * l2_loss
        else:
            self.loss = tf.reduce_mean(-log_ll)

    def predict(self, x, seq_len, chars, char_seq_len):
        feed_dict = self._get_feed_dict(x, seq_len, chars, char_seq_len, is_train=False)
        logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=feed_dict)
        y_pred = viterbi_decode(logits, trans_params, seq_len - 1)
        return y_pred

    def compute_accuracy(self, x, seq_len, chars, char_seq_len, y):
        return np.mean(np.equal(self.predict(x, seq_len, chars, char_seq_len), y).astype(np.float32))

    def _build_train_op(self):
        with tf.variable_scope('train_step'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.lr)
            if self.cfg.grad_clip is not None:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)  # clip by global gradient norm
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, train_set, batch_size, epochs, texts):
        self.logger.info('Start training...')
        prev_f1 = 0.0
        # init_lr = self.lr
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('Epoch %2d/%2d:' % (epoch, epochs))
            n_batches = (len(train_set) + batch_size - 1) // batch_size
            prog = Progbar(target=n_batches)  # nbatches
            total_cost = 0
            total_samples = 0
            for i, (x, seq_len, chars, char_seq_len, y) in enumerate(batch_iter(train_set, batch_size, shuffle=True)):
                feed_dict = self._get_feed_dict(x, seq_len, chars, char_seq_len, y, is_train=True)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                total_cost += train_loss
                total_samples += x.shape[0]
                prog.update(i + 1, [("Train Loss", train_loss), ("Perplexity", np.exp(total_cost / total_samples))])
            # evaluate after one epoch finished
            f1s = []
            for text in texts:
                f1, *_ = self.compute_score(text, text + '.out')
                f1s.append(f1)
            if f1s[0] is not nan and f1s[0] >= prev_f1:
                prev_f1 = f1s[0]
                self.save_session(epoch)

    def compute_score(self, dataset_file, output_path):
        with codecs.open(dataset_file, 'r', encoding='utf-8') as f:
            text = f.read().split()
        text = [w for w in text if w not in self.cfg.punc_vocab and w not in PUNCTUATION_MAPPING] + [END]
        index = 0
        MAX_SUBSEQUENCE_LEN = 200
        with codecs.open(output_path, 'w', encoding='utf-8') as f_out:
            while True:
                subsequence = text[index: index + MAX_SUBSEQUENCE_LEN]
                if len(subsequence) == 0:
                    break
                # create feed data
                converted_seq = np.array([[self.cfg.word_vocab.get(w, self.cfg.word_vocab[UNK]) for w in subsequence]],
                                         dtype=np.int32)
                seq_len = np.array([len(v) for v in converted_seq], dtype=np.int32)
                converted_seq_chars = []
                for w in subsequence:
                    chars = [self.cfg.char_vocab.get(c, self.cfg.char_vocab[UNK]) for c in w]
                    converted_seq_chars.append(chars)
                converted_seq_chars, seq_len_char = pad_sequences([converted_seq_chars], pad_tok=0, nlevels=2)
                converted_seq_chars = np.array(converted_seq_chars, dtype=np.int32)
                seq_len_char = np.array(seq_len_char, dtype=np.int32)
                # predict
                y = self.predict(converted_seq, seq_len, converted_seq_chars, seq_len_char)
                # write to file
                f_out.write(subsequence[0])
                last_eos_idx = 0
                punctuations = []
                for y_t in y[0]:
                    punctuation = self.cfg.rev_punc_vocab[y_t]
                    punctuations.append(punctuation)
                    if punctuation in EOS_TOKENS:
                        last_eos_idx = len(punctuations)
                if subsequence[-1] == END:
                    step = len(subsequence) - 1
                elif last_eos_idx != 0:
                    step = last_eos_idx
                else:
                    step = len(subsequence) - 1
                for j in range(step):
                    f_out.write(' ' + punctuations[j] + ' ' if punctuations[j] != SPACE else ' ')
                    if j < step - 1:
                        f_out.write(subsequence[1 + j])
                if subsequence[-1] == END:
                    break
                index += step
        out_str, f1, err, ser = compute_error(dataset_file, output_path)
        self.logger.info('\nevaluate on {}:'.format(dataset_file))
        self.logger.info(out_str + '\n')
        try:  # delete output file after compute scores
            os.remove(output_path)
        except OSError:
            pass
        return f1, err, ser

    def evaluate(self, dataset, batch_size, epoch, step):
        acc = []
        total_cost = 0
        total_samples = 0
        print('\n')
        for (x, seq_len, chars, char_seq_len, y) in tqdm(batch_iter(dataset, batch_size, shuffle=False)):
            feed_dict = self._get_feed_dict(x, seq_len, chars, char_seq_len, y, is_train=False)
            val_loss = self.sess.run(self.loss, feed_dict=feed_dict)
            accuracy = self.compute_accuracy(x, seq_len, chars, char_seq_len, y)
            acc.append(accuracy)
            total_cost += val_loss
            total_samples += x.shape[0]
        accuracy = np.mean(acc)
        ppl = np.exp(total_cost / total_samples)
        self.logger.info('evaluate at epoch {}, step {} -- Accuracy: {}, Perplexity: {}'.format(epoch, step, accuracy,
                                                                                                ppl))
        return accuracy, ppl
