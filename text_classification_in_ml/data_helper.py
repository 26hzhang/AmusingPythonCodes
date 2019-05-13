import re
import os
import jieba
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_chinese_stopwords(filename):
    stopword_list = []
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            stopword_list.append(line)
    return stopword_list


def load_chinese_data(file_path, save_path, test_size=0.1, verbose=True):
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, sep=",", header=0)
    else:
        data = pd.read_excel(file_path, sheet_name="sheet1")
        data = data.rename(index=str, columns={"分类": "label", "正文": "text"})

        # tokenization
        jieba.enable_parallel(16)
        data["tokens"] = data["text"].apply(lambda x: jieba.cut(x.strip()))
        data["tokens"] = [" ".join(x) for x in data["tokens"]]
        data["tokens"] = data["tokens"].apply(
            lambda x: re.sub(" +", " ", x.strip().replace("\n", " ").replace("\t", " ")))
        data.to_csv(save_path, sep=",", header=True, index=False, na_rep="")

    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(data.label.values)

    x_train, x_test, y_train, y_test = train_test_split(data.tokens.values, labels, stratify=labels, random_state=1234,
                                                        test_size=test_size, shuffle=True)

    if verbose:
        print("sample tokenized text: {}".format(data["tokens"].values[0]), flush=True)
        print("labels: {}".format(data.label.unique()), flush=True)
        print("train set shape: {}, test set shape: {}".format(x_train.shape, x_test.shape))

    return x_train, x_test, y_train, y_test
