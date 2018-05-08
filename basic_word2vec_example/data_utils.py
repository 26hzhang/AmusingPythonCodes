import os
from six.moves import urllib
import zipfile
import tensorflow as tf
import collections
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    url = 'http://mattmahoney.net/dc/text8.zip'
    folder = "data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    local_filename = os.path.join(folder, filename)
    if not os.path.exists(local_filename):
        print('Download text8.zip...')
        local_filename, _ = urllib.request.urlretrieve(url, local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')
    return local_filename


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    # data - list of codes (integers from 0 to vocabulary_size-1).
    #   This is the original text but words are replaced by their codes
    # count - map of words(strings) to count of occurences
    # dictionary - map of words(strings) to their codes(integers)
    # reverse_dictionary - maps codes(integers) to words(strings)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(data, data_index, batch_sz, n_skips, skip_wd):
    assert batch_sz % n_skips == 0
    assert n_skips <= 2 * skip_wd
    batch = np.ndarray(shape=batch_sz, dtype=np.int32)
    labels = np.ndarray(shape=(batch_sz, 1), dtype=np.int32)
    span = 2 * skip_wd + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index: data_index + span])
    data_index += span
    for i in range(batch_sz // n_skips):
        context_words = [w for w in range(span) if w != skip_wd]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words)
        for j in range(n_skips):
            batch[i * n_skips + j] = buffer[skip_wd]
            context_word = words_to_use.pop()
            labels[i * n_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, data_index


def tsne_and_plot(embeddings, labels, filename='./data/tsne.png'):
    """TSNE dimension reduction and plot"""
    # Function to draw visualization of distance between embeddings.
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embeddings = tsne.fit_transform(embeddings)
    assert low_dim_embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)
    plt.show()
