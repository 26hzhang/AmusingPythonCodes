import csv
import numpy as np
import pickle as pkl


SMS_FILENAME = 'data/sms/sms.txt'
MADURAI_FILENAME = 'data/madurai/sample.txt'
KERNEL_FILENAME = 'data/linux/linux_kernel_3Maa'
PAULG_FILENAME = 'data/paulg/paulg.txt'

MADURAI_PATH = 'data/madurai/'
SMS_PATH = 'data/sms/'
KERNEL_PATH = 'data/linux/'
PAULG_PATH = 'data/paulg/'


def read_lines_sms(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return [row[-1] for row in list(reader)]


def read_lines(filename):
    with open(filename, 'r') as f:
        return f.read().split('\n')


def index_(lines):
    vocab = list(set('\n'.join(lines)))
    ch2idx = {k: v for v, k in enumerate(vocab)}
    return vocab, ch2idx


def to_array(lines, seqlen, ch2idx):
    # combine into one string
    raw_data = '\n'.join(lines)
    num_chars = len(raw_data)
    # calc data_len
    data_len = num_chars//seqlen
    # create numpy arrays
    x = np.zeros([data_len, seqlen])
    y = np.zeros([data_len, seqlen])
    # fill in
    for i in range(0, data_len):
        x[i] = np.array([ch2idx[ch] for ch in raw_data[i * seqlen: (i + 1) * seqlen]])
        y[i] = np.array([ch2idx[ch] for ch in raw_data[(i * seqlen) + 1: ((i + 1) * seqlen) + 1]])
    return x.astype(np.int32), y.astype(np.int32)


def process_data(path, filename, seqlen=20):
    lines = read_lines(filename)
    idx2ch, ch2idx = index_(lines)
    x, y = to_array(lines, seqlen, ch2idx)
    np.save(path + 'idx_x.npy', x)
    np.save(path + 'idx_y.npy', y)
    with open(path + 'metadata.pkl', 'wb') as f:
        pkl.dump({'idx2ch': idx2ch, 'ch2idx': ch2idx}, f)


def load_data(path):
    # read data control dictionaries
    with open(path + 'metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    # read numpy arrays
    x = np.load(path + 'idx_x.npy')
    y = np.load(path + 'idx_y.npy')
    return x, y, metadata['idx2ch'], metadata['ch2idx']


if __name__ == '__main__':
    process_data(path=PAULG_PATH, filename=PAULG_FILENAME)
