from rnn_units.vanilla_rnn import VanillaRNN
from data.paulg import datagen
import os
from data import data_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# process data (if the paulg folder doesn't contain '.npy' and '.pkl' data)
# datagen.process_data('./data/paulg/', './data/paulg/paulg.txt', seqlen=20)
# print('data processing, done...')

# load dataset
x, y, idx2ch, ch2idx = datagen.load_data('./data/paulg/')

# hyperparameters
hsize = 256
num_classes = len(idx2ch)
seq_len = x.shape[1]
state_size = 256
batch_size = 128

# create model
vanilla_rnn_model = VanillaRNN(num_classes, state_size, seq_len)

# training
train_set = data_utils.rand_batch_gen(x, y, batch_size=batch_size)
vanilla_rnn_model.train(train_set, epochs=50, steps_per_epoch=500)

# generate
text = vanilla_rnn_model.generate(idx2ch, ch2idx, num_words=100, separator='')
print(text)
