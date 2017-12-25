from rnn_units.stacked_lstm_rnn import StackedLstmRNN
from data.paulg import datagen
from data import data_utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# process data (if the paulg folder doesn't contain '.npy' and '.pkl' data)
# datagen.process_data('./data/paulg/', './data/paulg/paulg.txt', seqlen=20)
# print('data processing, done...')

# load data
x, y, idx2w, w2idx = datagen.load_data('data/paulg/')

stacked_lstm_rnn_model = StackedLstmRNN(num_classes=len(idx2w), state_size=512, num_layers=3, learning_rate=0.05)

# train model
train_set = data_utils.rand_batch_gen(x, y, batch_size=128)
stacked_lstm_rnn_model.train(train_set, epochs=1000, steps_per_epoch=100)

# generate text
text = stacked_lstm_rnn_model.generate(idx2w, w2idx, num_words=100, separator='')
print(text)
