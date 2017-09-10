import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import preprocessing
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

data = pd.read_csv("prices-split-adjusted.csv", index_col=0)
data['adj close'] = data.close
data.drop(['close'], 1, inplace=True)
data = data[data.symbol == 'GOOG']
data.drop(['symbol'], 1, inplace=True)

def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
    return df

df = normalize_data(data)

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]

    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]

    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
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

window = 22
x_train, y_train, x_test, y_test = load_data(df, window)

model = build_model([5, window, 1])
model.fit(x_train, y_train, batch_size=512, epochs=90, validation_split=0.1, verbose=1)

diff = []
ratio = []
p = model.predict(x_test)
print(p.shape)

for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append(y_test[u]/pr - 1)
    diff.append(abs(y_test[u] - pr))


def denormalize(data, normalized_value):
    data = data['adj close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(data)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

newp = denormalize(data, p)
newy_test = denormalize(data, y_test)

def model_score(model, x_train, y_train, x_test, y_test):
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], sqrt(trainScore[0])))

    testScore = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], sqrt(testScore[0])))
    return trainScore[0], testScore[0]

model_score(model, x_train, y_train, x_test, y_test)

plt.plot(newp, color='red', label='Prediction')
plt.plot(newy_test, color='blue', label='Actual')
plt.legend(loc='best')
plt.show()
