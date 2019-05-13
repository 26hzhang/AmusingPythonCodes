import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.feature_extraction import text


def one_hot_encoding(arr, num_class):
    result = np.zeros(shape=[arr.shape[0], num_class])
    for i, val in enumerate(arr):
        result[i, val] = 1
    return result


def multiclass_logloss(labels, predicts, eps=1e-12):
    """Multi-class log loss
    :param labels: labels
    :param predicts: predictions
    :param eps: epsilon
    :return: loss
    """
    if len(labels.shape) == 1:
        labels = one_hot_encoding(labels, predicts.shape[1])
    clip = np.clip(predicts, a_min=eps, a_max=1.0 - eps)
    vsota = np.sum(labels * np.log(clip))
    log_loss = -1.0 / labels.shape[0] * vsota
    label_idx = np.argmax(labels, axis=-1)
    predicts_idx = np.argmax(predicts, axis=-1)
    accuracy = np.mean(np.equal(label_idx, predicts_idx), dtype=np.float32) * 100
    return log_loss, accuracy


def number_normalizer(tokens):
    return ("[NUM]" if tokens[0].isdigit() else token for token in tokens)


class TfIdfVectorizer(text.TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(TfIdfVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def tfidf_vectorization(x_train, x_test, stop_words, min_df=3, max_df=0.5, max_features=None, ngram_range=(1, 2),
                        use_idf=True, smooth_idf=True):
    tfidf_vectorizer = TfIdfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, ngram_range=ngram_range,
                                       use_idf=use_idf, smooth_idf=smooth_idf, stop_words=stop_words)
    tfidf_vectorizer.fit(list(x_train) + list(x_test))
    x_train_tfv = tfidf_vectorizer.transform(x_train)
    x_test_tfv = tfidf_vectorizer.transform(x_test)
    return x_train_tfv, x_test_tfv


def word_count_vectorization(x_train, x_test, stop_words, min_df=3, max_df=0.5, ngram_range=(1, 2)):
    count_vectorizer = text.CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range,
                                            stop_words=stop_words)
    count_vectorizer.fit(list(x_train) + list(x_test))
    x_train_ctv = count_vectorizer.transform(x_train)
    x_test_ctv = count_vectorizer.transform(x_test)
    return x_train_ctv, x_test_ctv


def svd_decomposition(x_train, x_test, n_components=120, algorithm="randomized", n_iter=5):
    # SVD decomposition
    svd = decomposition.TruncatedSVD(n_components=n_components, algorithm=algorithm, n_iter=n_iter)
    svd.fit(x_train)
    x_train_svd = svd.transform(x_train)
    x_test_svd = svd.transform(x_test)
    # data standardize
    scl = preprocessing.StandardScaler()
    scl.fit(x_train_svd)
    x_train_svd_scl = scl.transform(x_train_svd)
    x_test_svd_scl = scl.transform(x_test_svd)
    return x_train_svd_scl, x_test_svd_scl
