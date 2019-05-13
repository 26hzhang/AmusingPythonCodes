from .model.gan_model import GAN
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# load data
mnist = input_data.read_data_sets("./data/MNIST_data", one_hot=True)

# create model
input_size = 784  # 28*28 pixels
gen_hidden_size = 256
disc_hidden_size = 256
noise_size = 100  # Noise data points
gan = GAN(input_size, gen_hidden_size, disc_hidden_size, noise_size)

# training
# sess = gan.train(mnist, num_steps=70000, batch_size=128, sess=None)

# restore model
sess = gan.restore_last_session()

# generate
# gan.generate(sess=sess)
gan.generate2(sess=sess)
