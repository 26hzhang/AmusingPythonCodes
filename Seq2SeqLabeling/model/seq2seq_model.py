import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper, ResidualWrapper
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention, AttentionWrapper, TrainingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode, GreedyEmbeddingHelper
from ..utils.data_utils import GO, EOS, batchnize_dataset
from ..utils.logger import Progbar
from ..utils.CoNLLeval import CoNLLeval


class SequenceToSequence:
    def __init__(self, config, mode="train", resume_training=False, model_name="Seq2SeqModel"):
        self.cfg = config
        self.ce = CoNLLeval()
        self.mode = mode
        self.resume_training, self.start_epoch = resume_training, 1
        self.model_name = model_name
        self.use_beam_search = False  # used only for decode mode
        if self.mode == "decode" and self.cfg.use_beam_search and self.cfg.beam_size > 1:
            self.use_beam_search = True
        ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        if self.mode == "decode" and not (ckpt and ckpt.model_checkpoint_path):
            print("WARNINGS: Setting `decode` mode but no checkpoint found in \"{}\" path, train a new model instead..."
                  .format(self.cfg.ckpt_path))
            self.mode = "train"
            self.use_beam_search = False
        self.sess, self.saver = None, None
        self.merged_summaries, self.summary_writer = None, None
        self._add_placeholders()
        self._build_model()
        if self.mode == "train":
            self._build_loss_op()
            self._build_train_op()
        self.cfg.logger.info("number of trainable parameters: {}.".format(np.sum([np.prod(v.get_shape().as_list())
                                                                                  for v in tf.trainable_variables()])))
        self.initialize_session()

    def initialize_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)
        if self.mode == "train" and self.resume_training:
            if not (ckpt and ckpt.model_checkpoint_path):
                print("No checkpoint found in directory %s. Start a new training session..." % self.cfg.ckpt_path)
                return
            r = input("Checkpoint found in {}. Do you want to resume this training session?\n(y)es | (n)o : "
                      .format(self.cfg.ckpt_path))
            if r.startswith("y") or r.startswith("Y"):
                ckpt_path = ckpt.model_checkpoint_path
                self.start_epoch = int(ckpt_path.split("-")[-1]) + 1
                print("Resuming training from {}, start epoch: {}".format(self.cfg.ckpt_path, self.start_epoch))
                self.saver.restore(self.sess, ckpt_path)

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        else:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, step):
        self.saver.save(self.sess, self.cfg.ckpt_path + self.model_name, global_step=step)

    def close_session(self):
        self.sess.close()

    def reinitialize_weights(self, scope_name=None):
        if scope_name is not None:  # reinitialize weights in the given scope name
            variables = tf.contrib.framework.get_variables(scope_name)
        else:  # reinitialize all weights
            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        self.sess.run(tf.variables_initializer(variables))

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.cfg.summary_dir + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.cfg.summary_dir + "test")

    def _add_placeholders(self):
        # shape = (batch_size, max_words_len)
        self.enc_source = tf.placeholder(dtype=tf.int32, shape=[None, None], name="encoder_input")
        self.dec_target_in = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_input")
        self.dec_target_out = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_output")
        # shape = (batch_size, )
        self.enc_seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="encoder_seq_length")
        self.dec_seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="decoder_seq_length")
        # hyper-parameters
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        self.lr = tf.placeholder(dtype=tf.float32, name="learning_rate")

    def _get_feed_dict(self, batch_data, keep_prob=None, lr=None):
        feed_dict = {self.enc_source: batch_data["words"], self.enc_seq_len: batch_data["source_len"],
                     self.dec_target_in: batch_data["tags_in"], self.dec_target_out: batch_data["tags_out"],
                     self.dec_seq_len: batch_data["target_len"], self.batch_size: batch_data["batch_size"]}
        if keep_prob is not None:
            feed_dict[self.keep_prob] = keep_prob
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _create_rnn_cell(self):
        cell = GRUCell(self.cfg.num_units) if self.cfg.cell_type == "gru" else LSTMCell(self.cfg.num_units)
        if self.cfg.use_dropout:
            cell = DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        if self.cfg.use_residual:
            cell = ResidualWrapper(cell)
        return cell

    def _create_encoder_cell(self):
        return MultiRNNCell([self._create_rnn_cell() for _ in range(self.cfg.num_layers)])

    def _create_decoder_cell(self):
        enc_outputs, enc_states, enc_seq_len = self.enc_outputs, self.enc_states, self.enc_seq_len
        batch_size = self.batch_size * self.cfg.beam_size if self.use_beam_search else self.batch_size
        with tf.variable_scope("attention"):
            if self.cfg.attention == "luong":  # Luong attention mechanism
                attention_mechanism = LuongAttention(num_units=self.cfg.num_units, memory=enc_outputs,
                                                     memory_sequence_length=enc_seq_len)
            else:  # default using Bahdanau attention mechanism
                attention_mechanism = BahdanauAttention(num_units=self.cfg.num_units, memory=enc_outputs,
                                                        memory_sequence_length=enc_seq_len)

        def cell_input_fn(inputs, attention):  # define cell input function to keep input/output dimension same
            # reference: https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/AttentionWrapper
            if not self.cfg.use_attention_input_feeding:
                return inputs
            input_project = tf.layers.Dense(self.cfg.num_units, dtype=tf.float32, name='attn_input_feeding')
            return input_project(tf.concat([inputs, attention], axis=-1))

        if self.cfg.top_attention:  # apply attention mechanism only on the top decoder layer
            cells = [self._create_rnn_cell() for _ in range(self.cfg.num_layers)]
            cells[-1] = AttentionWrapper(cells[-1], attention_mechanism=attention_mechanism, name="Attention_Wrapper",
                                         attention_layer_size=self.cfg.num_units, initial_cell_state=enc_states[-1],
                                         cell_input_fn=cell_input_fn)
            initial_state = [state for state in enc_states]
            initial_state[-1] = cells[-1].zero_state(batch_size=batch_size, dtype=tf.float32)
            dec_init_states = tuple(initial_state)
            cells = MultiRNNCell(cells)
        else:
            cells = MultiRNNCell([self._create_rnn_cell() for _ in range(self.cfg.num_layers)])
            cells = AttentionWrapper(cells, attention_mechanism=attention_mechanism, name="Attention_Wrapper",
                                     attention_layer_size=self.cfg.num_units, initial_cell_state=enc_states,
                                     cell_input_fn=cell_input_fn)
            dec_init_states = cells.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_states)
        return cells, dec_init_states

    def _build_model(self):
        with tf.variable_scope("embeddings"):
            self.source_embs = tf.get_variable(name="source_embs", shape=[self.cfg.source_vocab_size, self.cfg.emb_dim],
                                               dtype=tf.float32, trainable=True)
            self.target_embs = tf.get_variable(name="embeddings", shape=[self.cfg.vocab_size, self.cfg.emb_dim],
                                               dtype=tf.float32, trainable=True)
            source_emb = tf.nn.embedding_lookup(self.source_embs, self.enc_source)
            target_emb = tf.nn.embedding_lookup(self.target_embs, self.dec_target_in)
            print("source embedding shape: {}".format(source_emb.get_shape().as_list()))
            print("target input embedding shape: {}".format(target_emb.get_shape().as_list()))

        with tf.variable_scope("encoder"):
            if self.cfg.use_bi_rnn:
                with tf.variable_scope("bi-directional_rnn"):
                    cell_fw = GRUCell(self.cfg.num_units) if self.cfg.cell_type == "gru" else \
                        LSTMCell(self.cfg.num_units)
                    cell_bw = GRUCell(self.cfg.num_units) if self.cfg.cell_type == "gru" else \
                        LSTMCell(self.cfg.num_units)
                    bi_outputs, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, source_emb, dtype=tf.float32,
                                                              sequence_length=self.enc_seq_len)
                    source_emb = tf.concat(bi_outputs, axis=-1)
                    print("bi-directional rnn output shape: {}".format(source_emb.get_shape().as_list()))
            input_project = tf.layers.Dense(units=self.cfg.num_units, dtype=tf.float32, name="input_projection")
            source_emb = input_project(source_emb)
            print("encoder input projection shape: {}".format(source_emb.get_shape().as_list()))
            enc_cells = self._create_encoder_cell()
            self.enc_outputs, self.enc_states = dynamic_rnn(enc_cells, source_emb, sequence_length=self.enc_seq_len,
                                                            dtype=tf.float32)
            print("encoder output shape: {}".format(self.enc_outputs.get_shape().as_list()))

        with tf.variable_scope("decoder"):
            self.max_dec_seq_len = tf.reduce_max(self.dec_seq_len, name="max_dec_seq_len")
            self.dec_cells, self.dec_init_states = self._create_decoder_cell()
            # define input and output projection layer
            input_project = tf.layers.Dense(units=self.cfg.num_units, name="input_projection")
            self.dense_layer = tf.layers.Dense(units=self.cfg.vocab_size, name="output_projection")
            if self.mode == "train":  # either "train" or "decode"
                # for training
                target_emb = input_project(target_emb)
                train_helper = TrainingHelper(target_emb, sequence_length=self.dec_seq_len, name="train_helper")
                train_decoder = BasicDecoder(self.dec_cells, helper=train_helper, output_layer=self.dense_layer,
                                             initial_state=self.dec_init_states)
                self.dec_output, _, _ = dynamic_decode(train_decoder, impute_finished=True,
                                                       maximum_iterations=self.max_dec_seq_len)
                print("decoder output shape: {} (vocab size)".format(self.dec_output.rnn_output.get_shape().as_list()))

                # for decode
                start_token = tf.ones(shape=[self.batch_size, ], dtype=tf.int32) * self.cfg.target_dict[GO]
                end_token = self.cfg.target_dict[EOS]

                def inputs_project(inputs):
                    return input_project(tf.nn.embedding_lookup(self.target_embs, inputs))

                dec_helper = GreedyEmbeddingHelper(embedding=inputs_project, start_tokens=start_token,
                                                   end_token=end_token)
                infer_decoder = BasicDecoder(self.dec_cells, helper=dec_helper, initial_state=self.dec_init_states,
                                             output_layer=self.dense_layer)
                infer_dec_output, _, _ = dynamic_decode(infer_decoder, maximum_iterations=self.cfg.maximum_iterations)
                self.dec_predicts = infer_dec_output.sample_id

    def _build_loss_op(self):
        dec_logits = tf.identity(self.dec_output.rnn_output)
        dec_mask = tf.sequence_mask(self.dec_seq_len, self.max_dec_seq_len, dtype=tf.float32, name="dec_mask")
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=dec_logits, targets=self.dec_target_out, weights=dec_mask)
        tf.summary.scalar("loss", self.loss)

    def _build_train_op(self):
        with tf.variable_scope("train_step"):
            if self.cfg.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.cfg.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.cfg.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.cfg.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            else:  # default adam optimizer
                if self.cfg.optimizer != 'adam':
                    print('Unsupported optimizing method {}. Using default adam optimizer.'.format(self.cfg.optimizer))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, train_data, valid_set, test_set, valid_data, shuffle=True):
        self.cfg.logger.info("Start training...")
        self._add_summary()
        cur_step = 0
        best_f1 = -np.inf
        no_imprv_epoch = 0
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            self.cfg.logger.info("Epoch {} / {}:".format(epoch, self.cfg.epochs))
            if shuffle:
                random.shuffle(train_data)
            train_set = batchnize_dataset(train_data, self.cfg.source_dict, self.cfg.target_dict, self.cfg.batch_size)
            num_batches = len(train_set)
            prog = Progbar(target=num_batches)  # nbatches
            for i, batch_data in enumerate(train_set):
                cur_step += 1
                feed_dict = self._get_feed_dict(batch_data, keep_prob=self.cfg.keep_prob, lr=self.cfg.lr)
                _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
                prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", loss)])
                self.train_writer.add_summary(summary, (epoch - 1) * num_batches + i)
                if i % 100 == 0:
                    valid_feed_dict = self._get_feed_dict(valid_data, keep_prob=1.0, lr=None)
                    valid_summary = self.sess.run(self.summary, feed_dict=valid_feed_dict)
                    self.test_writer.add_summary(valid_summary, (epoch - 1) * num_batches + i)

            valid_score = self.evaluate(valid_set, "valid", epoch)
            self.cfg.logger.info("Evaluate on valid dataset, accuracy: {:04.2f}, precision: {:04.2f}, "
                                 "recall: {:04.2f}, FB1: {:04.2f}".format(valid_score["accuracy"],
                                                                          valid_score["precision"],
                                                                          valid_score["recall"], valid_score["FB1"]))

            test_score = self.evaluate(test_set, "test", epoch)
            self.cfg.logger.info("Evaluate on test dataset, accuracy: {:04.2f}, precision: {:04.2f}, "
                                 "recall: {:04.2f}, FB1: {:04.2f}".format(test_score["accuracy"],
                                                                          test_score["precision"],
                                                                          test_score["recall"], test_score["FB1"]))

            if test_score["FB1"] > best_f1:
                best_f1 = test_score["FB1"]
                no_imprv_epoch = 0
                self.save_session(epoch)
                self.cfg.logger.info(' -- new BEST score on test dataset: {:04.2f}'.format(best_f1))
            else:
                no_imprv_epoch += 1
                if no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.cfg.logger.info('early stop at {}th epoch without improve for {} epochs, BEST score: {:04.2f}'
                                         .format(epoch, no_imprv_epoch, best_f1))
                    break

        self.cfg.logger.info("Training process done...")
        self.train_writer.close()
        self.test_writer.close()

    def evaluate(self, dataset, data_name, epoch):
        save_path = os.path.join(self.cfg.predict_path, "{}-{}.txt".format(data_name, epoch))
        predictions = list()
        groundtruth = list()
        words_list = list()
        for data in dataset:
            feed_dict = {self.enc_source: data["words"], self.enc_seq_len: data["source_len"],
                         self.batch_size: data["batch_size"], self.keep_prob: 1.0}
            predicts = self.sess.run(self.dec_predicts, feed_dict=feed_dict)
            for tags, preds, words, seq_len in zip(data["tags_out"], predicts, data["words"], data["source_len"]):
                tags = [self.cfg.rev_target_dict[x] for x in tags[:seq_len]]
                preds = [self.cfg.rev_target_dict[x] for x in preds[:seq_len]]
                words = [self.cfg.rev_source_dict[x] for x in words[:seq_len]]
                predictions.append(preds)
                groundtruth.append(tags)
                words_list.append(words)
        score = self.ce.conlleval(predictions, groundtruth, words_list, save_path)
        return score
