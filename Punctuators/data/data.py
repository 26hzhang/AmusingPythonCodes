import os
import operator
import numpy as np
import codecs
import fnmatch
import re

DATA_PATH = "./data"
home = os.path.expanduser('~')
PRETRAINED_EMBEDDINGS_PATH = os.path.join(home, 'utilities', 'embeddings', 'glove.840B.300d.txt')

END = "</S>"
UNK = "<UNK>"
NUM = "<NUM>"
PAD = "<PAD>"

SPACE = "_SPACE"

MAX_WORD_VOCABULARY_SIZE = 100000
MIN_WORD_COUNT_IN_VOCAB = 1
MIN_CHAR_COUNT_IN_VOCAB = 10
MAX_SEQUENCE_LEN = 200

TRAIN_FILE = os.path.join(DATA_PATH, "train.txt")
DEV_FILE = os.path.join(DATA_PATH, "dev.txt")
REF_FILE = os.path.join(DATA_PATH, "ref.txt")
ASR_FILE = os.path.join(DATA_PATH, "asr.txt")

WORD_VOCAB_FILE = os.path.join(DATA_PATH, "vocabulary")
CHAR_VOCAB_FILE = os.path.join(DATA_PATH, 'char_vocabulary')

# Comma, period & question mark only:
PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"]
PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ",COMMA", ";SEMICOLON": ".PERIOD", "-DASH": ",COMMA"}

EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}
# punctuations that are not included in vocabulary nor mapping, must be added to CRAP_TOKENS
CRAP_TOKENS = {"<doc>", "<doc.>"}

numbers = re.compile(r"\d")
is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6


# replacement for pickling that takes less RAM. Useful for large datasets.
def dump(d, path):
    with open(path, 'w') as f:
        for s in d:
            f.write("%s\n" % repr(s))


def load(path):
    d = []
    with open(path, 'r') as f:
        for l in f:
            d.append(eval(l))
    return d


def add_counts(word_counts, char_counts, line):
    for w in line.split():
        if w in CRAP_TOKENS or w in PUNCTUATION_VOCABULARY or w in PUNCTUATION_MAPPING:
            continue
        if is_number(w):
            word_counts[NUM] = word_counts.get(NUM, 0) + 1
            continue
        word_counts[w] = word_counts.get(w, 0) + 1
        for c in w:
            char_counts[c] = char_counts.get(c, 0) + 1


def build_vocabulary(word_counts):
    # UNK and NUM will be appended to end
    print('word counts size: {}'.format(len(word_counts)))
    return [wc[0] for wc in reversed(sorted(word_counts.items(), key=operator.itemgetter(1))) if
            wc[1] >= MIN_WORD_COUNT_IN_VOCAB and wc[0] != UNK and wc[0] != NUM][:MAX_WORD_VOCABULARY_SIZE]


def build_char_vocab(char_counts):
    return [cc[0] for cc in reversed(sorted(char_counts.items(), key=operator.itemgetter(1))) if
            cc[1] >= MIN_CHAR_COUNT_IN_VOCAB and cc[0] != UNK]


def write_vocabulary(vocabulary, file_name):
    if NUM not in vocabulary:
        vocabulary.append(NUM)
    if END not in vocabulary:
        vocabulary.append(END)
    if UNK not in vocabulary:
        vocabulary.append(UNK)
    print("Vocabulary size: %d" % len(vocabulary))
    with codecs.open(file_name, mode='w', encoding='utf-8') as f:
        f.write("\n".join(vocabulary))


def write_char_vocab(char_vocab, filename):
    if END not in char_vocab:
        char_vocab.append(END)
    if UNK not in char_vocab:
        char_vocab.append(UNK)
    if PAD not in char_vocab:
        char_vocab = [PAD] + char_vocab
    print("Char vocabulary size: %d" % len(char_vocab))
    with codecs.open(filename, mode='w', encoding='utf-8') as f:
        f.write('\n'.join(char_vocab))


def load_glove_vocab(filename):
    with open(filename, 'r') as f:
        vocab = {line.strip().split()[0] for line in f}
    return vocab


def iterable_to_dict(arr):
    return dict((x.strip(), i) for (i, x) in enumerate(arr))


def read_vocabulary(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f:
        return iterable_to_dict(f.readlines())


def filter_and_save_glove_vectors(vocab, glove_path, save_path, dim):
    print('Filtering {} dim embeddings...'.format(dim))
    scale = np.sqrt(3.0 / dim)
    embeddings = np.random.uniform(-scale, scale, [len(vocab), dim])
    mask = np.zeros([len(vocab)])
    with open(glove_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                mask[word_idx] = 1
                embeddings[word_idx] = np.asarray(embedding)
            elif word.lower() in vocab and mask[vocab[word.lower()]] == 0:  # since tokens in train sets are lowercase
                word = word.lower()
                word_idx = vocab[word]
                mask[word_idx] = 1
                embeddings[word_idx] = np.asarray(embedding)
    print("Embeddings shape: {}".format(embeddings.shape))
    np.savez_compressed(save_path, embeddings=embeddings)


def write_processed_dataset(input_files, output_file):
    """
    data will consist of two sets of aligned sub-sequences (words and punctuations) of MAX_SEQUENCE_LEN tokens
    (actually punctuation sequence will be 1 element shorter).
    If a sentence is cut, then it will be added to next subsequence entirely
    (words before the cut belong to both sequences)
    """
    data = []
    word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
    punctuation_vocabulary = iterable_to_dict(PUNCTUATION_VOCABULARY)
    char_vocabulary = read_vocabulary(CHAR_VOCAB_FILE)
    num_total = 0
    num_unks = 0
    current_words = []
    current_chars = []
    current_punctuations = []
    last_eos_idx = 0  # if it's still 0 when MAX_SEQUENCE_LEN is reached, then the sentence is too long and skipped.
    last_token_was_punctuation = True  # skip first token if it's punctuation
    # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence
    skip_until_eos = False
    for input_file in input_files:
        with codecs.open(input_file, 'r', encoding='utf-8') as text:
            for line in text:
                for token in line.split():
                    # First map oov punctuations to known punctuations
                    if token in PUNCTUATION_MAPPING:
                        token = PUNCTUATION_MAPPING[token]
                    if skip_until_eos:
                        if token in EOS_TOKENS:
                            skip_until_eos = False
                        continue
                    elif token in CRAP_TOKENS:
                        continue
                    elif token in punctuation_vocabulary:
                        # if we encounter sequences like: "... !EXLAMATIONMARK .PERIOD ...",
                        # then we only use the first punctuation and skip the ones that follow
                        if last_token_was_punctuation:
                            continue
                        if token in EOS_TOKENS:
                            last_eos_idx = len(current_punctuations)  # no -1, because the token is not added yet
                        punctuation = punctuation_vocabulary[token]
                        current_punctuations.append(punctuation)
                        last_token_was_punctuation = True
                    else:
                        if not last_token_was_punctuation:
                            current_punctuations.append(punctuation_vocabulary[SPACE])
                        chars = []
                        for c in token:
                            c = char_vocabulary.get(c, char_vocabulary[UNK])
                            chars.append(c)
                        if is_number(token):
                            token = NUM
                        word = word_vocabulary.get(token, word_vocabulary[UNK])
                        current_words.append(word)
                        current_chars.append(chars)
                        last_token_was_punctuation = False
                        num_total += 1
                        num_unks += int(word == word_vocabulary[UNK])
                    if len(current_words) == MAX_SEQUENCE_LEN:  # this also means, that last token was a word
                        assert len(current_words) == len(current_punctuations) + 1, \
                            "#words: %d; #punctuations: %d" % (len(current_words), len(current_punctuations))
                        # Sentence did not fit into subsequence - skip it
                        if last_eos_idx == 0:
                            skip_until_eos = True
                            current_words = []
                            current_chars = []
                            current_punctuations = []
                            # next sequence starts with a new sentence, so is preceded by eos which is punctuation
                            last_token_was_punctuation = True
                        else:
                            subsequence = [
                                current_words[:-1] + [word_vocabulary[END]],
                                current_chars[:-1] + [[char_vocabulary[END]]],
                                current_punctuations,
                            ]
                            data.append(subsequence)
                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx + 1:]
                            current_chars = current_chars[last_eos_idx + 1:]
                            current_punctuations = current_punctuations[last_eos_idx + 1:]
                        last_eos_idx = 0  # sequence always starts with a new sentence
    print("%.2f%% UNK-s in %s" % (num_unks / num_total * 100, output_file))
    dump(data, output_file)


def create_datasets(root_path, train_out, dev_out, ref_out, asr_out, embeddings_path=None):
    train_files = []
    dev_files = []
    ref_files = []
    asr_files = []

    word_counts = dict()
    char_counts = dict()

    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, '*.txt'):
            path = os.path.join(root, filename)
            if filename.endswith("ref.txt"):
                ref_files.append(path)
            elif filename.endswith("asr.txt"):
                asr_files.append(path)
            elif filename.endswith("dev.txt"):
                dev_files.append(path)
            else:
                train_files.append(path)
                with codecs.open(path, 'r', 'utf-8') as text:
                    for line in text:
                        add_counts(word_counts, char_counts, line)

    # build char vocabulary
    char_vocabulary = build_char_vocab(char_counts)
    print("Len of char vocabulary: {}".format(len(char_vocabulary)))
    write_char_vocab(char_vocabulary, CHAR_VOCAB_FILE)

    # build word vocabulary
    vocabulary = build_vocabulary(word_counts)
    print("Len of word vocabulary: {}".format(len(vocabulary)))
    glove_vocab = load_glove_vocab(embeddings_path)
    glove_vocab = glove_vocab & {word.lower() for word in glove_vocab}
    print("Len of glove vocabulary: {}".format(len(glove_vocab)))
    vocabulary = [word for word in vocabulary if word in glove_vocab]
    print("Len of word vocabulary after filtered: {}".format(len(vocabulary)))
    write_vocabulary(vocabulary, WORD_VOCAB_FILE)
    if embeddings_path:
        vocabulary = read_vocabulary(WORD_VOCAB_FILE)
        del vocabulary[NUM], vocabulary[END], vocabulary[UNK]  # delete special tokens when build pre-trained embeddings
        filtered_emb_path = os.path.join(DATA_PATH, 'glove.840B.300d.filtered.npz')
        filter_and_save_glove_vectors(vocabulary, embeddings_path, filtered_emb_path, dim=300)

    write_processed_dataset(train_files, train_out)
    write_processed_dataset(dev_files, dev_out)
    write_processed_dataset(ref_files, ref_out)
    write_processed_dataset(asr_files, asr_out)


def main():
    """if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        sys.exit("The path to the source data directory with txt files is missing")"""
    create_datasets("./data/raw/", TRAIN_FILE, DEV_FILE, REF_FILE, ASR_FILE, PRETRAINED_EMBEDDINGS_PATH)


if __name__ == "__main__":
    main()
