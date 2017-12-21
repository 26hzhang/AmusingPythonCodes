import matplotlib.pyplot as plt
import pandas as pd
from stock_prices_predict.data_utils import normalize_data, load_data
from stock_prices_predict.model import lstm_rnn_model, denormalize, model_score
import warnings
import os

# suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# read data
data = pd.read_csv("prices-split-adjusted.csv", index_col=0)
data['adj close'] = data.close
data.drop(['close'], 1, inplace=True)
data = data[data.symbol == 'GOOG']
data.drop(['symbol'], 1, inplace=True)

# normalize data
df = normalize_data(data)

# load data
window = 22
x_train, y_train, x_test, y_test = load_data(df, seq_len=window)

# create model and fit dataset
model = lstm_rnn_model([5, window, 1])
model.fit(x_train, y_train, batch_size=512, epochs=90, validation_split=0.1, verbose=1)

diff = []
ratio = []
p = model.predict(x_test)  # make prediction
print(p.shape)

for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append(y_test[u]/pr - 1)
    diff.append(abs(y_test[u] - pr))

# compute train and test score
model_score(model, x_train, y_train, x_test, y_test)

newp = denormalize(data, p)
newy_test = denormalize(data, y_test)

plt.plot(newp, color='red', label='Prediction')
plt.plot(newy_test, color='blue', label='Actual')
plt.legend(loc='best')
plt.show()
