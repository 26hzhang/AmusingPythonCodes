from model.dcgan_model import DCGANBatchNorm
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# load data
mnist = input_data.read_data_sets("./data/MNIST_data", one_hot=True)

# create model
image_dim = 784  # 28*28 pixels * 1 channel
noise_dim = 100  # Noise data points
batch_size = 128
dcgan_batch_model = DCGANBatchNorm(image_dim, noise_dim, batch_size=batch_size)

# training
# sess = dcgan_batch_model.train(mnist, num_steps=10000, sess=None)

# restore
sess = dcgan_batch_model.restore_last_session()

# generate
dcgan_batch_model.generate(sess)
