from ..data.data import WORD_VOCAB_FILE, PUNCTUATION_VOCABULARY, CHAR_VOCAB_FILE
from ..data.data import read_vocabulary, iterable_to_dict
from ..utils.data_utils import load_embeddings
import os


class Config:
    def __init__(self):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.word_vocab = read_vocabulary(WORD_VOCAB_FILE)
        self.vocab_size = len(self.word_vocab)
        self.char_vocab = read_vocabulary(CHAR_VOCAB_FILE)
        self.char_vocab_size = len(self.char_vocab)
        self.punc_vocab = iterable_to_dict(PUNCTUATION_VOCABULARY)
        self.label_size = len(self.punc_vocab)
        self.rev_word_vocab = {v: k for k, v in self.word_vocab.items()}
        self.rev_punc_vocab = {v: k for k, v in self.punc_vocab.items()}
        self.embedding = load_embeddings(os.path.join('.', 'data', 'glove.840B.300d.filtered.npz'))

    # word emb parameters
    emb_dim = 300

    # char emb parameters
    char_emb = None
    char_emb_dim = 50
    char_num_units = 50

    # model parameters
    num_units = 300

    # for dense connected lstm layers
    num_layers = 4
    num_units_list = [50, 50, 50, 300]

    # train.txt parameters
    l2_reg = None
    grad_clip = 2.0
    lr = 0.001
    lr_decay = 0.05
    keep_prob = 0.75
    resume_training = False
    ckpt_path = './ckpt/'
    model_name = "punctuator"
    model_save_path = ckpt_path + model_name
