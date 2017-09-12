# Amusing Python Codes
Interesting python codes to deal with small tasks

## House Prices Predict
It's a [Kaggle](https://www.kaggle.com/) competition · [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Some useful references**:

[[详解 Kaggle 房价预测竞赛优胜方案：用 Python 进行全面数据探索]](https://www.leiphone.com/news/201704/Py7Mu3TwRF97pWc7.html), [[PCA and Regression]](https://www.kaggle.com/miguelangelnieto/pca-and-regression), [[How to get to TOP 25% with Simple Model (sklearn)]](https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn), [[XGBoost+Lasso]](https://www.kaggle.com/humananalog/xgboost-lasso/code/code).

## Stock Prices Predict
It's a small dataset of stock prices in [Kaggle](https://www.kaggle.com/) · [New York Stock Exchange](https://www.kaggle.com/dgawlik/nyse).

**Some useful references**:

[[Predict stock prices with LSTM]](https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm), [[LSTM_Stock_prediction-20170507]](https://www.kaggle.com/benjibb/lstm-stock-prediction-20170507), [[Using F score to evaluate the LSTM model]](https://www.kaggle.com/amberhahn/using-f-score-to-evaluate-the-lstm-model/code), [[BenjiKCF/Neural-Network-with-Financial-Time-Series-Data]](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data).

## WeChat Exploration using itchat Package
Using itchat package to login your wechat, extracting information of your friends and analyzing to obtain some interesting results.

itchat package api: [[link]](https://itchat.readthedocs.io/zh/latest/api/).

Some analysis by myself: [[link]](https://isaacchanghau.github.io/2017/09/10/Python-itchat包分析微信朋友/).

## Douban · Wolf Warrior II Film Comments Analysis
Using python requests and re package to wirte a [douban](https://movie.douban.com/subject/26363254/comments?start=0) web crawler to extract 190000+ comments data, and using jieba, weodcloud, pandas and so on to do simple analysis and get some useful results.

Some analysis by myself: [[link]](https://isaacchanghau.github.io/2017/09/10/Python-浅析-战狼2-170000-影评数据/).

## Basic Word2Vec Example
The codes are from tensorflow github page: [word2vec_basic.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py). It helps to understand the theory of word2vec as well as learn to use tensorflow to build your own program.

## MNIST Dataset Training Examples
This folder contains five iPython notebooks and two python codes.

1. Simple MNIST classification with logistic regression. It's a practice to learn the MNIST dataset and use a simple regressor to classify the dataset.
2. MNIST Training, saving and loading model. It's a practice to learn how to save and load models trained by tensorflow. Reference: [10_save_restore_nrt.py](https://github.com/nlintz/TensorFlow-Tutorials/blob/master/10_save_restore_net.py).
3. Visualize training process. A practice to learn how to visualize the training process with tensorboard. Reference: [mnist_with_summaries.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py).
4. Train MNIST dataset via CNN. A simple convolutional neural networks example to deal with a classification tasks. Reference: [05_convolutional_net.py](https://github.com/nlintz/TensorFlow-Tutorials/blob/master/05_convolutional_net.py).
5. Train MNIST dataset via RNN. A simple recurrent neural networks example to train MNIST dataset. Reference: [recurrent_network.py](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py), [Supervised Sequence Labelling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf).
6. Train MNIST with autoencoder. An example of unsupervised learning. Reference: [Autoencoders and Sparsity](http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity).

Reference book: [Tensorflow 技术解析与实战](http://www.epubit.com.cn/book/details/4862).