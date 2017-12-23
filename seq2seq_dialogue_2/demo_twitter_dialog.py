import seq2seq_wrapper
from datasets.twitter import data
import data_utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024


model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len, yseq_len=yseq_len, xvocab_size=xvocab_size, yvocab_size=yvocab_size,
                                ckpt_path='ckpt/twitter/', emb_dim=emb_dim, num_layers=3)

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

model.train(train_batch_gen, val_batch_gen)

sess = model.restore_last_session()

# TODO
