from .data_helper import load_chinese_data
from .data_helper import load_chinese_stopwords
from .utils import multiclass_logloss
from .utils import tfidf_vectorization
from .utils import word_count_vectorization
from .utils import svd_decomposition
from .classifier import logistic_regression
from .classifier import naive_bayes
from .classifier import svm
from .classifier import xgboost_classifier

"""
Reference: 
https://zhuanlan.zhihu.com/p/64602471
https://github.com/goto456/stopwords
"""

chinese_stopwords = load_chinese_stopwords("./datasets/中文停用词表.txt")

x_train, x_test, y_train, y_test = load_chinese_data(file_path="./datasets/复旦大学中文文本分类语料.xlsx",
                                                     save_path="./datasets/fudan.csv",
                                                     test_size=0.1,
                                                     verbose=True)

# TF-IDF Vectorizer
print("run TF-IDF vectorization,", end=" ", flush=True)
x_train_tfv, x_test_tfv = tfidf_vectorization(x_train, x_test, chinese_stopwords, min_df=3, max_df=0.5,
                                              max_features=None, ngram_range=(1, 2), use_idf=True, smooth_idf=True)
print("done...", flush=True)


# Count Vectorizer
print("run WordCount vectorization,", end=" ", flush=True)
x_train_ctv, x_test_ctv = word_count_vectorization(x_train, x_test, chinese_stopwords, min_df=3, max_df=0.5,
                                                   ngram_range=(1, 2))
print("done...", flush=True)


# SVD decomposition and standard scaler
print("run SVD decomposition and standard scaler,", end=" ", flush=True)
x_train_tfv_svd, x_test_tfv_svd = svd_decomposition(x_train_tfv, x_test_tfv, n_components=120, algorithm="randomized",
                                                    n_iter=5)
print("done...", flush=True)


# logistic regression + TF-IDF
print("run logistic regression (with TF-IDF),", end=" ", flush=True)
predictions = logistic_regression(x_train=x_train_tfv, x_test=x_test_tfv, y_train=y_train, solver="lbfgs", tol=1e-4,
                                  reg_strength=1.0, penalty="l2", multi_class="multinomial")
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("logistic regression (with TF-IDF), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# logistic regression + Word Count
print("run logistic regression (with WordCount),", end=" ", flush=True)
predictions = logistic_regression(x_train=x_train_ctv, x_test=x_test_ctv, y_train=y_train, solver="lbfgs", tol=1e-4,
                                  reg_strength=1.0, penalty="l2", multi_class="multinomial")
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("logistic regression (with WordCount), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# naive bayes + TF-IDF
print("run naive bayes (with TF-IDF),", end=" ", flush=True)
predictions = naive_bayes(x_train=x_train_tfv, x_test=x_test_tfv, y_train=y_train, alpha=1.0, fit_prior=True)
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("naive bayes (with TF-IDF), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# naive bayes + Word Count
print("run naive bayes (with WordCount),", end=" ", flush=True)
predictions = naive_bayes(x_train=x_train_ctv, x_test=x_test_ctv, y_train=y_train, alpha=1.0, fit_prior=True)
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("naive bayes (with WordCount), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# TF-IDF feature + SVD decomposition + SVM
print("run SVM (with TF-IDF + SVD),", end=" ", flush=True)
predictions = svm(x_train=x_train_tfv_svd, x_test=x_test_tfv_svd, y_train=y_train, penalty_param=1.0,
                  kernel="rbf", degree=3, gamma="auto_deprecated", coef0=0.0, shrinking=True, probability=True,
                  tol=1e-4)
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("SVM (with TF-IDF + SVD), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# XGBoost + TF-IDF
print("run XGBoost (with TF-IDF),", end=" ", flush=True)
predictions = xgboost_classifier(x_train=x_train_tfv, x_test=x_test_tfv, y_train=y_train, max_depth=7, n_estimators=200,
                                 colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1, tocsc=True)
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("XGBoost (with TF-IDF), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# XGBoost + Word Count
print("run XGBoost (with WordCount),", end=" ", flush=True)
predictions = xgboost_classifier(x_train=x_train_ctv, x_test=x_test_ctv, y_train=y_train, max_depth=7, n_estimators=200,
                                 colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1, tocsc=True)
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("XGBoost (with WordCount), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))


# TF-IDF feature + SVD decomposition + XGBoost
print("run XGBoost (with TF-IDF + SVD),", end=" ", flush=True)
predictions = xgboost_classifier(x_train=x_train_tfv_svd, x_test=x_test_tfv_svd, y_train=y_train, max_depth=7,
                                 n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1,
                                 tocsc=False)
log_loss, accuracy = multiclass_logloss(labels=y_test, predicts=predictions, eps=1e-12)
print("done...", flush=True)
print("XGBoost (with TF-IDF + SVD), logloss: %0.3f, accuracy: %0.3f" % (log_loss, accuracy))

'''
run TF-IDF vectorization, done...
run WordCount vectorization, done...
run SVD decomposition and standard scaler, done...

run logistic regression (with TF-IDF), done...
logistic regression (with TF-IDF), logloss: 0.550, accuracy: 90.270

run logistic regression (with WordCount), done...
logistic regression (with WordCount), logloss: 0.407, accuracy: 93.622

run naive bayes (with TF-IDF), done...
naive bayes (with TF-IDF), logloss: 1.005, accuracy: 75.243

run naive bayes (with WordCount), done...
naive bayes (with WordCount), logloss: 3.196, accuracy: 87.568

run SVM (with TF-IDF + SVD), done...
SVM (with TF-IDF + SVD), logloss: 0.274, accuracy: 91.135

run XGBoost (with TF-IDF), done...
XGBoost (with TF-IDF), logloss: 0.140, accuracy: 95.892

run XGBoost (with WordCount), done...
XGBoost (with WordCount), logloss: 0.131, accuracy: 96.108

run XGBoost (with TF-IDF + SVD), done...
XGBoost (with TF-IDF + SVD), logloss: 0.251, accuracy: 92.108
'''
