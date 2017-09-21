# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.translate import data_utils

"""
This tutorial refers: http://github.com/suriyadeepan/easy_seq2seq
However, it is no longer maintained, since the author creates another repository named "practical_seq2seq",
and does experiments in that repo. Here is the link: https://github.com/suriyadeepan/practical_seq2seq
Anyway, we will still use the codes in easy_seq2seq as example.
Dataset available here: https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus
"""


class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, max_gradient_norm, batch_size,
                 learning_rate, learning_rate_dacay_factor, use_lstm=False, num_samples=512, forward_only=False):
        """build model
        parameters:
            source_vocab_size: the vocabulary size of query words
            tagert_vocab_size: the vocabulary size of answer words
            buckets: (I,O), I means the maximal length of input sentence, O means the maximal length of output sentences
            size: the number of neurons of each layer
            num_layers: the number of layers
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of each batch. The batch_size for training and predicting can be different
            learning_rate: learning rate
            learning_rate_decay_factor: the factor to control the learning rate
            use_lstm: default is GRU, change to LSTM by setting it True
            num_samples: the number of samples to use softmax
            forward_only: build one direction propagation
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_dacay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None

        # if the size of samples is smaller than the size of vocabulary, then using softmax to do sampling
        if 0 < num_samples < self.target_vocab_size:
            w = tf.get_variable('proj_w', [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b', [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)

            softmax_loss_function = sampled_loss

        # construct RNN
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)

        cell = single_cell
        if num_layers:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # Attention model
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                                             num_encoder_symbols=source_vocab_size,
                                                             num_decoder_symbols=target_vocab_size,
                                                             embedding_size=size, output_projection=output_projection,
                                                             feed_previous=do_decode)

        # feed data for model
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weight{0}'.format(i)))

        # the value of targets is one status skewing of the decoder (targets的值是解码器偏移一位)
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # the output of training model
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                                                         targets, self.target_weights, buckets,
                                                                         lambda x, y: seq2seq_f(x, y, True),
                                                                         softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                                                         targets, self.target_weights, buckets,
                                                                         lambda x, y: seq2seq_f(x, y, False),
                                                                         softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.AdamOptimizer()
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.
        Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            decoder_inputs: list of numpy int vectors to feed as decoder inputs.
            target_weights: list of numpy float vectors to feed as target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward step or only forward.
        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.
        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError('Encoder length must be equal to the one in bucket, %d != %d.' % (len(encoder_inputs),
                                                                                               encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError('Decoder length must be equal to the one in bucket, %d != %d.' % (len(decoder_inputs),
                                                                                               decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError('Weight length must be equal to the one in bucket, %d != %d.' % (len(target_weights),
                                                                                             decoder_size))

        # feed inputs
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # output feed
        if not forward_only:
            output_feed = [self.updates[bucket_id], self.gradient_norms[bucket_id], self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            return outputs[1], outputs[2], None  # 有后向传播下的输出：梯度，损失值，None
        else:
            return None, outputs[0], outputs[1:]  # 仅有前向传播下的输出：None，损失值，outputs

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
            data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
            bucket_id: integer, which bucket to get the batch for.
        Returns:
            The triple (encoder_inputs, decoder_inputs, target_weights) for
            the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data, pad them if needed, reverse encoder inputs and
        # add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                                  for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                                                  for batch_idx in range(self.batch_size)], dtype=np.int32))
            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol. The corresponding target is decoder_
                # input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
