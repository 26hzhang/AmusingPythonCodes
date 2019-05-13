# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
# from tensorflow.models.rnn.translate import data_utils
from .data_utils import PAD_ID, GO_ID

"""
This tutorial refers: http://github.com/suriyadeepan/easy_seq2seq
However, it is no longer maintained, since the author creates another repository named "practical_seq2seq",
and does experiments in that repo. Here is the link: https://github.com/suriyadeepan/practical_seq2seq
Anyway, we will still use the codes in easy_seq2seq as example.
Dataset available here: https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus
Tensorflow URL: https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate
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
        self.buckets = buckets  # (I,O), I: max length of input sentence, O: max length of output sentences
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
                # sampled_softmax_loss computes and returns the sampled softmax training loss
                # url: https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss
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
            """url: https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq
               url: https://www.tensorflow.org/tutorials/recurrent
            This model first embeds encoder_inputs by a newly created embedding (of shape
                [num_encoder_symbols x input_size]).
            Then it runs an RNN to encode embedded encoder_inputs into a state vector. It keeps the outputs of this RNN
                at every step to use for attention later.
            Next, it embeds decoder_inputs by another newly created embedding (of shape
                [num_decoder_symbols x input_size]).
            Then it runs attention decoder, initialized with the last encoder state, on embedded decoder_inputs and
                attending to encoder outputs.
            Params:
                encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
                decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
                cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
                num_encoder_symbols: Integer; number of symbols on the encoder side.
                num_decoder_symbols: Integer; number of symbols on the decoder side.
                embedding_size: Integer, the length of the embedding vector for each symbol.
                num_heads: Number of attention heads that read from attention_states.
                output_projection: None or a pair (W, B) of output projection weights and biases; W has shape
                         [output_size x num_decoder_symbols] and B has shape [num_decoder_symbols]; if provided and
                         feed_previous=True, each fed previous output will first be multiplied by W and added B.
                feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of decoder_inputs will be used
                         (the "GO" symbol), and all other decoder inputs will be taken from previous outputs (as in
                         embedding_rnn_decoder). If False, decoder_inputs are used as given (the standard decoder case).
                dtype: The dtype of the initial RNN state (default: tf.float32).
                scope: VariableScope for the created subgraph; defaults to "embedding_attention_seq2seq".
                initial_state_attention: If False (default), initial attentions are zero. If True, initialize the
                         attentions from the initial state and attention states.
            Returns:
                A tuple of the form (outputs, state), where: outputs: A list of the same length as decoder_inputs of 2D
                         Tensors with shape [batch_size x num_decoder_symbols] containing the generated outputs.
                         state: The state of each decoder cell at the final time-step. It is a 2D Tensor of shape
                         [batch_size x cell.state_size].
            """
            # if do_decode == True, only the first of decoder_inputs will be used (the "GO" symbol), otherwise, decoder_
            # inputs are used, thus, for training, set do_decode as False, for testing, set as True
            return tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                             decoder_inputs,
                                                             cell,  # RNN cell
                                                             num_encoder_symbols=source_vocab_size,
                                                             num_decoder_symbols=target_vocab_size,
                                                             embedding_size=size,
                                                             output_projection=output_projection,
                                                             feed_previous=do_decode)

        # feed data for model
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))
        for i in range(buckets[-1][1] + 1):  # plus 1, since for decoder, the first symbol is <GO>
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weight{0}'.format(i)))

        # the value of targets is one status skewing of the decoder, ignore <GO> symbol 
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # the output of training model
        if forward_only:  # testing
            '''url: https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/model_with_buckets
            Create a sequence-to-sequence model with support for bucketing. The seq2seq argument is a function that 
            defines a sequence-to-sequence model,
                e.g., seq2seq = lambda x, y: basic_rnn_seq2seq( x, y, rnn_cell.GRUCell(24))
            Arguments:
                encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
                decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
                targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
                weights: List of 1D batch-sized float-Tensors to weight the targets.
                buckets: A list of pairs of (input size, output size) for each bucket.
                seq2seq: A sequence-to-sequence model function; it takes 2 input that agree with encoder_inputs and 
                         decoder_inputs, and returns a pair consisting of outputs and states (e.g., basic_rnn_seq2seq).
                softmax_loss_function: Function (labels, logits) -> loss-batch to be used instead of the standard 
                         softmax (the default if this is None). Note that to avoid confusion, it is required for the 
                         function to accept named arguments.
                per_example_loss: Boolean. If set, the returned loss will be a batch-sized tensor of losses for each 
                         sequence in the batch. If unset, it will be a scalar with the averaged loss from all examples.
                name: Optional name for this operation, defaults to "model_with_buckets".
            Returns:
                A tuple of the form (outputs, losses)
                outputs: The outputs for each bucket, its j'th element consist of a list of 2D Tensors. The shape of 
                         output tensors can be either [batch_size x output_size] or [batch_size x num_decoder_symbols] 
                         depending on the seq2seq model used. In our model its output tensors is 
                         [batch_size x num_decoder_symbols].
                losses: list of scalar Tensors, representing losses for each bucket, or, if per_example_loss is set, a 
                         list of 1D batch_sized float Tensors.
            '''
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs,
                                                                         self.decoder_inputs,
                                                                         targets,
                                                                         self.target_weights,
                                                                         buckets,
                                                                         # seq2seq model
                                                                         lambda x, y: seq2seq_f(x, y, True),
                                                                         softmax_loss_function=softmax_loss_function)
            if output_projection is not None:  # apply output projection
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:  # training
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs,
                                                                         self.decoder_inputs,
                                                                         targets,
                                                                         self.target_weights,
                                                                         buckets,
                                                                         lambda x, y: seq2seq_f(x, y, False),
                                                                         softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()  # get all the trainable variables
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.AdamOptimizer()
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                # http://blog.csdn.net/u013713117/article/details/56281715
                # https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
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
        input_feed = {}  # feed_dict: encoder_inputs, decoder_inputs, targets
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)  # _PAD

        # output feed: fetches, 
        if not forward_only:
            output_feed = [self.updates[bucket_id], self.gradient_norms[bucket_id], self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)  # runs operations and evaluates tensors in fetches

        if not forward_only:
            return outputs[1], outputs[2], None  # output for backward: gradient, loss, None
        else:
            return None, outputs[0], outputs[1:]  # output for forward_only: None, loss, outputs

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
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)  # ? no EOS symbol?

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
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
