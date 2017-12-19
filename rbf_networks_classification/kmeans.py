# encoding: utf-8
import tensorflow as tf

"""
KMeans Clustering
"""

class KMeans(object):
    """KMeans Clustering"""
    def __init__(self, k=20, max_iter=1000, session=None):
        self._k = k
        self._max_iter = max_iter
        self.sess = session or tf.Session()
        self.centers = None
        self.nearest = None

    def __random_init_center(self, data):
        """initializing the node randomly"""
        n_samples = tf.shape(data)[0]
        random_indices = tf.random_shuffle(tf.range(0, n_samples))
        begin = [0, ]
        size = [self._k, ]
        center_indices = tf.slice(random_indices, begin, size)
        centers = tf.gather(data, center_indices)
        return centers

    @staticmethod
    def __update_cluster(data, centers):
        """updating clusters"""
        expanded_data = tf.expand_dims(data, 0)
        expanded_center = tf.expand_dims(centers, 1)
        distance = tf.reduce_sum(tf.square(tf.subtract(expanded_data, expanded_center)), 2)
        # find the closest center for each node
        near_indices = tf.argmin(distance, 0)
        # current loss
        loss = tf.reduce_mean(tf.reduce_min(distance, 0))
        return near_indices, loss

    def __update_center(self, data, nearest):
        """updating centroid"""
        partitions = tf.dynamic_partition(data, tf.to_int32(nearest), self._k)
        # updating centers by means
        new_centers = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
        return new_centers

    def train(self, data):
        try:
            data = self.sess.run(data)
        except:
            data = tf.constant(data)
        initcenters = self.__random_init_center(data)
        # first iter
        nearest_indices, loss = self.__update_cluster(data, initcenters)
        centers = self.sess.run(self.__update_center(data, nearest_indices))
        lastloss = self.sess.run(loss)
        nearest = self.sess.run(nearest_indices)
        for i in range(self._max_iter):
            # updating nearest_indices (cluster) and orthocenter alternately
            nearest_indices, loss = self.__update_cluster(data, centers)
            centers = self.sess.run(self.__update_center(data, nearest_indices))
            lossvalue = self.sess.run(loss)
            nearest = self.sess.run(nearest_indices)
            print('iter:', i, 'loss:', lossvalue)
            if lastloss - lossvalue < 1e-8:
                break
            lastloss = lossvalue
        print('finished')
        self.centers = centers
        self.nearest = nearest
