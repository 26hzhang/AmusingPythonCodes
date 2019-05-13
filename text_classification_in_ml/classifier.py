from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb


def logistic_regression(x_train, x_test, y_train, penalty="l2", tol=1e-6, solver="lbfgs", multi_class="multinomial",
                        reg_strength=1.0):
    classifier = LogisticRegression(penalty=penalty, C=reg_strength, tol=tol, solver=solver, multi_class=multi_class)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict_proba(x_test)
    return predictions


def naive_bayes(x_train, x_test, y_train, alpha=1.0, fit_prior=True, class_prior=None):
    classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict_proba(x_test)
    return predictions


def svm(x_train, x_test, y_train, penalty_param=1.0, kernel="rbf", degree=3, gamma="auto_deprecated", coef0=0.0,
        shrinking=True, probability=True, tol=1e-4):
    classifier = SVC(C=penalty_param, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                     probability=probability, tol=tol)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict_proba(x_test)
    return predictions


def xgboost_classifier(x_train, x_test, y_train, max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8,
                       nthread=10, learning_rate=0.1, tocsc=True):
    classifier = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, colsample_bytree=colsample_bytree,
                                   subsample=subsample, nthread=nthread, learning_rate=learning_rate)
    if tocsc:
        classifier.fit(x_train.tocsc(), y_train)
        predictions = classifier.predict_proba(x_test.tocsc())
    else:
        classifier.fit(x_train, y_train)
        predictions = classifier.predict_proba(x_test)
    return predictions
