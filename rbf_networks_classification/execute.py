# encoding: utf-8
from rbf import RBFNet
import numpy as np
import scipy.io as scio
import warnings
warnings.filterwarnings('ignore')  # ignore all warnings

"""
This python file is the main function to execute the self-defined RBF network
to train the model and do measurement
"""

def obtainData():
    # load training data
    train_data = scio.loadmat('./data/data_train.mat')
    train_data = np.array(train_data['data_train'])
    # load training labels
    train_label = scio.loadmat('./data/label_train.mat')
    train_label = np.array(train_label['label_train'])
    # load testing data
    test_data = scio.loadmat('./data/data_test.mat')
    test_data = np.array(test_data['data_test'])
    return train_data, train_label, test_data


def main():
    train_data, train_label, test_data = obtainData()
    rbf = RBFNet(k=20, delta=0.2)  # build RBF networks
    rbf.fit(train_data, train_label)
    # obtain the weights after training
    weights = rbf._weights
    print('Weights:', np.array(weights).T[0])  # reshape and format
    # obtain accuracy of training dataset
    accuracy = rbf.accuracy()
    print('Accuracy:', accuracy)
    # make a prediction
    predicts = rbf.predict(test_data)
    # reshape and format
    print('Predictions:', np.array(predicts, dtype=np.int32).T[0])


if __name__ == "__main__":
    main()
