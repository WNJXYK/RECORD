from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def push(lX, ly, budget, X, y):
    ix = ly[:, 0] == y
    cur_X, cur_y = lX[ix], ly[ix]
    lX, ly = lX[~ix], ly[~ix]
    while cur_X.shape[0] >= budget:
        cur_X = cur_X[1:, :]
        cur_y = cur_y[1:, :]
    cur_X = np.vstack([cur_X, X.reshape(1, -1)])
    cur_y = np.vstack([cur_y, y.reshape(1, -1)])
    lX, ly = np.vstack([lX, cur_X]), np.vstack([ly, cur_y])

    return lX, ly


def generate_labels(n, c):
    return np.array([c] * n).reshape(-1, 1)


def random_item(X, m):
    if X.shape[0] < m: return X[np.random.choice(np.arange(X.shape[0]), m, replace=True)]
    return X[np.random.choice(np.arange(X.shape[0]), m, replace=False)]

class IFBinaryLogistic:
    def __init__(self):
        self.model = None
        self.scaler = None

    def sigmod(self, x):
        return 1. / (1 + np.exp(x))

    def train(self, X, y):
        C = 0.1 / X.shape[0]
        self.model = LogisticRegression(C=C, tol=1e-8, fit_intercept=False, solver='lbfgs', warm_start=True, max_iter=10000)
        self.model.fit(X, y)
        self.theta = self.model.coef_

    def get_influence(self, train_X, test_X, train_y, test_y):
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(np.vstack([train_X, test_X]))
        train_X = self.scaler.transform(train_X)
        test_X = self.scaler.transform(test_X)

        self.train(train_X, train_y)

        n, p = train_X.shape[0], train_X.shape[1]

        hessian = np.zeros((p, p))
        for i in range(int(n)):
            X, y = train_X[i].reshape((-1, 1)), train_y[i]
            hessian += self.sigmod(np.dot(self.theta, X)) * self.sigmod(-np.dot(self.theta, X)) * np.dot(X, X.transpose())
        hessian /= n
        ihessian = pinv(hessian)

        influence = np.zeros(n)
        test_X, test_y = test_X.transpose(), test_y
        for i in range(n):
            X, y = train_X[i].reshape((-1, 1)), train_y[i]
            partial_X = - self.sigmod(-y * np.dot(self.theta, X)) * y * X
            partial_test = - np.dot(np.multiply(test_y, self.sigmod(np.multiply(test_y, np.dot(self.theta, test_X)))), test_X.transpose())
            influence[i] = - np.dot(np.dot(partial_test, ihessian), partial_X)

        return influence

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return accuracy_score(self.predict(X), y)

def RECORDS_choice(lX, ly, X, pred, proba, m, m_last=None):
    if m_last is None: m_last = m
    proba = proba.max(axis=1)

    new_X, new_pred, new_proba = X[-m_last:], pred[-m_last:], proba[-m_last:]
    old_X, old_pred, old_proba = X[:-m_last], pred[:-m_last], proba[:-m_last]

    classes = list(set(list(new_pred)))
    class_budget = int(m / len(classes))

    prev_X, prev_y = np.zeros((0, X.shape[1])), np.zeros((0, 1))
    curr_X, curr_y = np.zeros((0, X.shape[1])), np.zeros((0, 1))
    curr_p = np.zeros((0, 1))
    for c in classes:
        ix = [new_pred[i] == c and new_proba[i] > 0.6 for i in range(new_X.shape[0])]
        n = int(np.sum(ix))
        sorted_X = new_X[ix][np.argsort(-new_proba[ix])]
        sorted_proba = new_proba[ix][np.argsort(-new_proba[ix])]
        div_point, div_margin = int(n / 2), 0
        prev_X, curr_X = np.vstack([prev_X, sorted_X[div_point:]]), np.vstack([curr_X, sorted_X[:div_point]])
        prev_y, curr_y = np.vstack([prev_y, generate_labels(n - div_point, c)]), np.vstack([curr_y, generate_labels(div_point, c)])
        curr_p = np.vstack([curr_p, sorted_proba[:div_point].reshape(-1, 1)])
    prev_y, curr_y = prev_y.ravel(), curr_y.ravel()

    for c in classes:
        n_tra, n_tst = np.sum(curr_y == c), np.sum(prev_y == c)
        if n_tra == 0 or n_tst == 0 or np.sum(curr_y != c) == 0 or np.sum(prev_y != c) == 0:
            sorted_X = np.vstack([curr_X[curr_y == c], prev_X[prev_y == c]])
            n = min(class_budget, n_tst + n_tra)
        else:
            tra_X = np.vstack([curr_X[curr_y == c], random_item(curr_X[curr_y != c], n_tra)])
            tst_X = np.vstack([prev_X[prev_y == c], random_item(prev_X[prev_y != c], n_tst)])
            tra_y = np.vstack([generate_labels(n_tra, 1), generate_labels(n_tra, -1)]).ravel()
            tst_y = np.vstack([generate_labels(n_tst, 1), generate_labels(n_tst, -1)]).ravel()

            clf = IFBinaryLogistic()
            influence = clf.get_influence(tra_X, tst_X, tra_y, tst_y)[: n_tra]
            # print(influence)
            sorted_X = curr_X[curr_y == c][np.argsort(influence)]
            sorted_proba = curr_p[curr_y == c][np.argsort(influence)]
            n = min(class_budget, min(n_tra, np.sum(influence < 0)))

        for i in range(n - 1, -1, -1):
            lX, ly = push(lX, ly, class_budget, sorted_X[i], c)

    return lX, ly, np.zeros((0, X.shape[1]))