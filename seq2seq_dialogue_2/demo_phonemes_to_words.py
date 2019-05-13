from .datasets.cmudict import data
from .data_utils import rand_batch_gen, decode, split_dataset
import importlib
from .seq2seq_wrapper import Seq2Seq
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# load data from pickle and npy files
data_ctl, idx_words, idx_phonemes = data.load_data(PATH='datasets/cmudict/')
(trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_phonemes, idx_words)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 128
xvocab_size = len(data_ctl['idx2pho'].keys())
yvocab_size = len(data_ctl['idx2alpha'].keys())
emb_dim = 128

# importlib.reload(seq2seq_wrapper)

model = Seq2Seq(xseq_len=xseq_len, yseq_len=yseq_len, xvocab_size=xvocab_size, yvocab_size=yvocab_size,
                ckpt_path='ckpt/cmudict/', emb_dim=emb_dim, num_layers=3)

val_batch_gen = rand_batch_gen(validX, validY, 16)
train_batch_gen = rand_batch_gen(trainX, trainY, 128)

model.train(train_batch_gen, val_batch_gen)

sess = model.restore_last_session()

output = model.predict(sess, val_batch_gen.__next__()[0])
print(output.shape)

for oi in output:
    print(decode(sequence=oi, lookup=data_ctl['idx2alpha'], separator=''))
