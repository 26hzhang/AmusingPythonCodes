# coding:utf-8
import tensorflow as tf
import sys
import ctypes
import numpy as np


class TransR(object):
    def __init__(self, entity_size, relation_size, hidden_size_e, hidden_size_r, margin=1.0, learning_rate=0.001,
                 l1_flag=True, model_name='transr_model', ckpt_path='./ckpt/transr/'):
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.hidden_size_e = hidden_size_e
        self.hidden_size_r = hidden_size_r
        self.margin = margin
        self.learning_rate = learning_rate
        self.l1_flag = l1_flag
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        # build graph
        sys.stdout.write('\nBuilding Graph...')
        tf.reset_default_graph()
        # set inputs
        # with tf.name_scope("read_inputs"):
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[self.entity_size, self.hidden_size_e],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[self.relation_size, self.hidden_size_r],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_matrix = tf.get_variable(name="rel_matrix", shape=[self.relation_size,
                                                                        self.hidden_size_e * self.hidden_size_r],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h), [-1, self.hidden_size_e, 1])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t), [-1, self.hidden_size_e, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r), [-1, self.hidden_size_r])
            neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h), [-1, self.hidden_size_e, 1])
            neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t), [-1, self.hidden_size_e, 1])
            neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r), [-1, self.hidden_size_r])
            matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_r),
                                [-1, self.hidden_size_r, self.hidden_size_e])

            pos_h_e = tf.reshape(tf.matmul(matrix, pos_h_e), [-1, self.hidden_size_r])
            pos_t_e = tf.reshape(tf.matmul(matrix, pos_t_e), [-1, self.hidden_size_r])
            neg_h_e = tf.reshape(tf.matmul(matrix, neg_h_e), [-1, self.hidden_size_r])
            neg_t_e = tf.reshape(tf.matmul(matrix, neg_t_e), [-1, self.hidden_size_r])

        if self.l1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        sys.stdout.write('Done...\n')

    def train(self, dataset, num_steps=3000, batch_size=100, sess=None):
        saver = tf.train.Saver()  # use to save the model
        if sess is None:
            sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        ph = np.zeros(batch_size, dtype=np.int32)
        pt = np.zeros(batch_size, dtype=np.int32)
        pr = np.zeros(batch_size, dtype=np.int32)
        nh = np.zeros(batch_size, dtype=np.int32)
        nt = np.zeros(batch_size, dtype=np.int32)
        nr = np.zeros(batch_size, dtype=np.int32)

        ph_addr = ph.__array_interface__['data'][0]
        pt_addr = pt.__array_interface__['data'][0]
        pr_addr = pr.__array_interface__['data'][0]
        nh_addr = nh.__array_interface__['data'][0]
        nt_addr = nt.__array_interface__['data'][0]
        nr_addr = nr.__array_interface__['data'][0]

        dataset.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        nbatches = dataset.getTripleTotal() // batch_size
        step = 0
        sys.stdout.write('Training started...\n')
        try:
            for step in range(1, num_steps + 1):
                res = 0
                for batch in range(nbatches):
                    dataset.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, batch_size)
                    feed_dict = {self.pos_h: ph, self.pos_t: pt, self.pos_r: pr, self.neg_h: nh, self.neg_t: nt,
                                 self.neg_r: nr}
                    _, _, loss = sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
                    res += loss
                print('  step %4d, loss: %8.4f' % (step, res / nbatches))
                if step % 100 == 0:  # save the model periodically
                    sys.stdout.write('Saving model at step %d... ' % step)
                    saver.save(sess, self.ckpt_path + self.model_name, global_step=step)
                    sys.stdout.write('Done...\n')
        except KeyboardInterrupt:
            sys.stdout.write('Interrupted by user at training step %d, saving model at this step.. ' % step)
        saver.save(sess, self.ckpt_path + self.model_name, global_step=step)
        sys.stdout.write('Done...\n')
        return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()  # create a session
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess

    def test(self, testset, sess=None):
        if sess is None:
            print('restore model from last check point...')
            sess = self.restore_last_session()
        batch_size = testset.getEntityTotal()

        ph = np.zeros(batch_size, dtype=np.int32)
        pt = np.zeros(batch_size, dtype=np.int32)
        pr = np.zeros(batch_size, dtype=np.int32)
        ph_addr = ph.__array_interface__['data'][0]
        pt_addr = pt.__array_interface__['data'][0]
        pr_addr = pr.__array_interface__['data'][0]

        testset.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        testset.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        testset.testHead.argtypes = [ctypes.c_void_p]
        testset.testTail.argtypes = [ctypes.c_void_p]

        num_steps = testset.getTestTotal()
        for step in range(num_steps):
            # test head
            testset.getHeadBatch(ph_addr, pt_addr, pr_addr)
            feed_dict = {self.pos_h: ph, self.pos_t: pt, self.pos_r: pr}
            _, predict_head = sess.run([self.global_step, self.predict], feed_dict)
            testset.testHead(predict_head.__array_interface__['data'][0])
            # test tail
            testset.getTailBatch(ph_addr, pt_addr, pr_addr)
            feed_dict = {self.pos_h: ph, self.pos_t: pt, self.pos_r: pr}
            _, predict_tail = sess.run([self.global_step, self.predict], feed_dict)
            testset.testTail(predict_tail.__array_interface__['data'][0])
            print(step)
            if step % 50 == 0:
                testset.test()
        testset.test()
