from model.dcgan_model import DCGAN
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# load data
mnist = input_data.read_data_sets("./data/MNIST_data", one_hot=True)

# create model
image_size = 784  # 28*28 pixels * 1 channel
noise_size = 200  # noise data points
dcgan = DCGAN(image_size, noise_size)

# training
# sess = dcgan.train(mnist, num_steps=10000, batch_size=64, sess=None)

# restore model
sess = dcgan.restore_last_session()

# generate
# dcgan.generate(sess=sess)
dcgan.generate2(sess=sess)
