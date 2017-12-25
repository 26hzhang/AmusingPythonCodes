from rnn_units.lstm_rnn import LstmRNN
from data.paulg import datagen
import os
from data import data_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# process data (if the paulg folder doesn't contain '.npy' and '.pkl' data)
# datagen.process_data('./data/paulg/', './data/paulg/paulg.txt', seqlen=20)
# print('data processing, done...')

# load data
x, y, idx2w, w2idx = datagen.load_data('data/paulg/')

lstm_rnn_model = LstmRNN(num_classes=len(idx2w), state_size=512)

# training
train_set = data_utils.rand_batch_gen(x, y, batch_size=256)
lstm_rnn_model.train(train_set, epochs=50, steps_per_epoch=100)

# generating
text = lstm_rnn_model.generate(idx2w, w2idx, num_words=100, separator='')
print(text)
