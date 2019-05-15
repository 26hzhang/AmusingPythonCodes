import ujson
import codecs
import random

GO = "<GO>"  # <s>: start of sentence
EOS = "<EOS>"  # </s>: end of sentence, also act as padding


def load_dataset(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = ujson.load(f)
    return dataset


def process_batch_data(batch_words, batch_tags, word_dict, tag_dict):
    b_words_len = [len(words) for words in batch_words]
    max_word_len = max(b_words_len)
    b_words, b_tags_in, b_tags_out = [], [], []
    for words, tags in zip(batch_words, batch_tags):
        assert len(words) == len(tags), "the size of words ({}) and tags ({}) doesn't match".format(
            len(words), len(tags))
        words = words + [word_dict[EOS]] * (max_word_len - len(words))
        tags_in = [tag_dict[GO]] + tags + [tag_dict[EOS]] * (max_word_len - len(tags))
        tags_out = tags + [tag_dict[EOS]] * (max_word_len - len(tags)) + [tag_dict[EOS]]
        b_words.append(words)
        b_tags_in.append(tags_in)
        b_tags_out.append(tags_out)
    b_tags_len = [x + 1 for x in b_words_len]
    return {"words": b_words, "tags_in": b_tags_in, "tags_out": b_tags_out, "source_len": b_words_len,
            "target_len": b_tags_len, "batch_size": len(b_words)}


def dataset_batch_iter(dataset, batch_size, word_dict, tag_dict):
    batch_words, batch_tags = [], []
    for record in dataset:
        batch_words.append(record["words"])
        batch_tags.append(record["tags"])
        if len(batch_words) == batch_size:
            yield process_batch_data(batch_words, batch_tags, word_dict, tag_dict)
            batch_words, batch_tags = [], []
    if len(batch_words) > 0:
        yield process_batch_data(batch_words, batch_tags, word_dict, tag_dict)


def batchnize_dataset(data, word_dict, tag_dict, batch_size=None, shuffle=True):
    if type(data) == str:
        dataset = load_dataset(data)
    else:
        dataset = data
    if shuffle:
        random.shuffle(dataset)
    batches = []
    if batch_size is None:
        for batch in dataset_batch_iter(dataset, len(dataset), word_dict, tag_dict):
            batches.append(batch)
        return batches[0]
    else:
        for batch in dataset_batch_iter(dataset, batch_size, word_dict, tag_dict):
            batches.append(batch)
        return batches
