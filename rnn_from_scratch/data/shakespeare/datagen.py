import nltk
import numpy as np
import pickle

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'shakespeare.txt'

VOCAB_SIZE = 12000

SEQ_LEN = 10


# read lines from file, return [list of lines]
def read_lines(filename):
    content = ''
    with open(filename) as f:
        for line in f:
            if line.strip():
                if not line.strip()[-1] == ':':
                    content += line
    return content.split('\n')[:-1]


# split sentences in one line into multiple lines, return [list of lines]
def split_line(line):
    return line.split('.')


# remove anything that isn't in the vocabulary, return str(pure ta/en)
def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


# read list of words, create index to word, word to index dictionaries, return tuple(vocab->(word, count), idx2w, w2idx)
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(tokenized_sentences)
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = [x[0] for x in vocab]
    # word2index
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


def to_array(tokenized, seqlen, w2idx):
    num_words = len(tokenized)
    # calc data_len
    data_len = num_words//seqlen
    # create numpy arrays
    x = np.zeros([data_len, seqlen])
    y = np.zeros([data_len, seqlen])
    # fill in
    for i in range(data_len):
        x[i] = np.array([w2idx[w] for w in tokenized[i * seqlen: (i + 1) * seqlen]])
        y[i] = np.array([w2idx[w] for w in tokenized[(i * seqlen) + 1: ((i + 1) * seqlen) + 1]])
    return x.astype(np.int32), y.astype(np.int32)


def process_data():
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)
    # change to lower case (just for en)
    lines = [line.lower() for line in lines]
    print('\n:: Sample from read(p) lines')
    print(lines[121: 125])
    # filter out unnecessary characters
    print('\n>> Filter lines')
    lines = [filter_line(line, EN_WHITELIST) for line in lines]
    print(lines[121:125])
    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    tokenized = [w for wordlist in lines for w in wordlist.split(' ')]
    print('\n:: Sample from segmented list of words')
    print(tokenized[60])
    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(tokenized, vocab_size=VOCAB_SIZE)
    # remove unknowns
    tokenized = [w for w in tokenized if w in idx2w]
    # convert to ndarray
    x, y = to_array(tokenized, SEQ_LEN, w2idx)
    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_x.npy', x)
    np.save('idx_y.npy', y)
    # let us now save the necessary dictionaries
    metadata = {'w2idx': w2idx, 'idx2w': idx2w, 'seqlen': SEQ_LEN, 'freq_dist': freq_dist}
    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_data(path=''):
    # read data control dictionaries
    with open(path + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_x = np.load(path + 'idx_x.npy')
    idx_y = np.load(path + 'idx_y.npy')
    return idx_x, idx_y, metadata['idx2w'], metadata['w2idx'], metadata['seqlen']


if __name__ == '__main__':
    process_data()
