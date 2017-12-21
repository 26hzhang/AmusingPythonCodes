from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
import time
from sklearn import preprocessing
from math import sqrt


def lstm_rnn_model(layers):
    model = Sequential()
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print("Compilation Time: ", time.time() - start)
    return model


def denormalize(data, normalized_value):
    data = data['adj close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(data)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


def model_score(model, x_train, y_train, x_test, y_test):
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (train_score[0], sqrt(train_score[0])))
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (test_score[0], sqrt(test_score[0])))
    return train_score[0], test_score[0]
