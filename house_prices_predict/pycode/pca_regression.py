import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import tflearn
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def lets_try(train, labels):
    results = {}

    def test_model(clf):
        cv = KFold(n_splits=5, shuffle=True, random_state=45)
        r2 = make_scorer(r2_score)
        r2_val_score = cross_val_score(clf, train, labels, cv=cv, scoring=r2)
        scores = [r2_val_score.mean()]
        return scores

    clf = linear_model.LinearRegression()
    results["Linear"] = test_model(clf)

    clf = linear_model.Ridge()
    results["Ridge"] = test_model(clf)

    clf = linear_model.BayesianRidge()
    results["Bayesian Ridge"] = test_model(clf)

    clf = linear_model.HuberRegressor()
    results["Hubber"] = test_model(clf)

    clf = linear_model.Lasso(alpha=1e-4)
    results["Lasso"] = test_model(clf)

    clf = BaggingRegressor()
    results["Bagging"] = test_model(clf)

    clf = RandomForestRegressor()
    results["RandomForest"] = test_model(clf)

    clf = AdaBoostRegressor()
    results["AdaBoost"] = test_model(clf)

    clf = svm.SVR()
    results["SVM RBF"] = test_model(clf)

    clf = svm.SVR(kernel="linear")
    results["SVM Linear"] = test_model(clf)

    results = pd.DataFrame.from_dict(results, orient='index')
    results.columns = ["R Square Score"]
    # results = results.sort(columns=["R Square Score"], ascending=False)
    results.plot(kind="bar", title="Model Scores")
    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    return results


'''
Pre-process is referred from: https://www.kaggle.com/miguelangelnieto/pca-and-regression/notebook/notebook
'''
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
labels = train["SalePrice"]
data = pd.concat([train, test], ignore_index=True)
print(data.shape)
print(data.dtypes.value_counts())
print(labels.describe())
print(data.head())
print(data.tail())

data = data.drop("SalePrice", 1)
ids = test['Id']
# Remove id and columns with more than a thousand missing values
data = data.drop("Id", 1)
data = data.drop("Alley", 1)
data = data.drop("Fence", 1)
data = data.drop("MiscFeature", 1)
data = data.drop("PoolQC", 1)
data = data.drop("FireplaceQu", 1)
# Count the column types
all_columns = data.columns.values
non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                   "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
                   "EnclosedPorch", "3SsnPorch", "ScreenPorch","PoolArea", "MiscVal"]
categorical = [value for value in all_columns if value not in non_categorical]
# One Hot Encoding and nan transformation
data = pd.get_dummies(data)
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
data = imp.fit_transform(data)
# log transform
data = np.log(data)
labels = np.log(labels)
data[data == -np.inf] = 0

results = lets_try(train, labels)
print(results)
plt.show()

pca = PCA(whiten=True)
pca.fit(data)
variance = pd.DataFrame(pca.explained_variance_ratio_)
var = np.cumsum(pca.explained_variance_ratio_)

# PCA
pca = PCA(n_components=36, whiten=True)
pca = pca.fit(data)
data_pca = pca.transform(data)

train = data_pca[:1460]
test = data_pca[1460:]

results = lets_try(train, labels)
print(results)
plt.show()

cv = KFold(n_splits=5, shuffle=True, random_state=45)
parameters = {'alpha': [1000, 100, 10], 'epsilon': [1.2, 1.25, 1.50], 'tol': [1e-10]}
clf = linear_model.HuberRegressor()
r2 = make_scorer(r2_score)
grid_obj = GridSearchCV(clf, parameters, cv=cv, scoring=r2)
grid_fit = grid_obj.fit(train, labels)
best_clf = grid_fit.best_estimator_
best_clf.fit(train, labels)

labels_nl = labels
labels_nl = labels_nl.reshape(-1, 1)

tf.reset_default_graph()
r2 = tflearn.R2()
net = tflearn.input_data(shape=[None, train.shape[1]])
net = tflearn.fully_connected(net, 30, activation='linear')
net = tflearn.fully_connected(net, 10, activation='linear')
net = tflearn.fully_connected(net, 1, activation='linear')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.01, decay_step=100)
net = tflearn.regression(net, optimizer=sgd, loss='mean_square', metric=r2)
model = tflearn.DNN(net)
model.fit(train, labels_nl, show_metric=True, validation_set=0.2, shuffle=True, n_epoch=50)

predictions_huber = best_clf.predict(test)
predictions_dnn = model.predict(test)
predictions_huber = np.exp(predictions_huber)
predictions_dnn = np.exp(predictions_dnn)
predictions_dnn = predictions_dnn.reshape(-1,)

sub = pd.DataFrame({"Id": ids, "SalePrice": predictions_dnn})
print(sub)
# Saving to CSV
# sub.to_csv('submission.csv', index=False)
