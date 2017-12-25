from rnn_units.gru_rnn import GruRNN
from data.paulg import datagen
from data import data_utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# process data (if the paulg folder doesn't contain '.npy' and '.pkl' data)
# datagen.process_data('./data/paulg/', './data/paulg/paulg.txt', seqlen=20)
# print('data processing, done...')

# load data
x, y, idx2w, w2idx = datagen.load_data('data/paulg/')

# create model
gru_rnn_model = GruRNN(num_classes=len(idx2w), state_size=256, learning_rate=0.1)

# training
train_set = data_utils.rand_batch_gen(x, y, batch_size=256)
gru_rnn_model.train(train_set)

# generate
text = gru_rnn_model.generate(idx2w, w2idx, num_words=100, separator='')
print(text)
