import codecs
import os
import ujson
from unicodedata import normalize
from collections import Counter

GO = "<GO>"  # <s>: start of sentence
EOS = "<EOS>"  # </s>: end of sentence, also act as padding
UNK = "<UNK>"  # for Unknown tokens
PAD = "<PAD>"  # padding not used


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def word_convert(word):
    # convert french characters to latin equivalents
    word = normalize("NFD", word).encode("ascii", "ignore").decode("utf-8")
    word = word.lower()
    return word


def raw_dataset_iter(filename):
    with codecs.open(filename, mode="r", encoding="cp1252") as f:
        words, tags = [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 and len(words) != 0:  # means read whole one sentence
                yield words, tags
                words, tags = [], []
            else:
                _, word, tag = line.split("\t")
                word = word_convert(word)
                words.append(word)
                tags.append(tag)


def load_dataset(filename):
    dataset = []
    for words, tags in raw_dataset_iter(filename):
        dataset.append({"words": words, "tags": tags})
    return dataset


def build_vocab(datasets):
    word_counter = Counter()
    tag_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
            tags = record["tags"]
            for tag in tags:
                tag_counter[tag] += 1
    word_vocab = [GO, EOS, UNK] + [word for word, _ in word_counter.most_common()]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    tag_vocab = [GO, EOS] + [tag for tag, _ in tag_counter.most_common()]
    tag_dict = dict([(tag, idx) for idx, tag in enumerate(tag_vocab)])
    return word_dict, tag_dict


def build_dataset(data, word_dict, tag_dict):
    dataset = []
    for record in data:
        words = [word_dict[word] if word in word_dict else word_dict[UNK] for word in record["words"]]
        tags = [tag_dict[tag] for tag in record["tags"]]
        dataset.append({"words": words, "tags": tags})
    return dataset


def process_data():
    # load raw data
    train_data = load_dataset(os.path.join("media", "train.crf"))
    dev_data = load_dataset(os.path.join("media", "dev.crf"))
    test_data = load_dataset(os.path.join("media", "test.crf"))
    # build vocabulary
    word_dict, _ = build_vocab([train_data, dev_data])
    _, tag_dict = build_vocab([train_data, dev_data, test_data])
    # create indices dataset
    train_set = build_dataset(train_data, word_dict, tag_dict)
    dev_set = build_dataset(dev_data, word_dict, tag_dict)
    test_set = build_dataset(test_data, word_dict, tag_dict)
    vocab = {"word_dict": word_dict, "tag_dict": tag_dict}
    # write to file
    write_json("vocab.json", vocab)
    write_json("train.json", train_set)
    write_json("dev.json", dev_set)
    write_json("test.json", test_set)


if __name__ == "__main__":
    process_data()
