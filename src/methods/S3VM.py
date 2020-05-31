import numpy as np
import sklearn.svm as svm

class S3VM(object):
    def __init__(self, kernel='linear', Cl=1.5, Cu=0.001, gamma="scale", C=1, max_iter = 1000):
        '''
        Initial TSVM

        Parameters
        ----------
        kernel: kernel of svm
        '''
        self.Cl, self.Cu = Cl, Cu
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.clf = svm.SVC(C=self.C, kernel=self.kernel, gamma=gamma, probability=True)

    def fit(self, X1, Y1, X2):
        '''
        Train TSVM by X1, Y1, X2

        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        '''
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1.ravel())
        if X2.shape[0] == 0: return

        Y2 = self.clf.predict(X2)
        Y2 = np.expand_dims(Y2, 1)
        Y1 = Y1.reshape((-1, 1))
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3.ravel(), sample_weight=sample_weight)
            iter = 0
            while iter < self.max_iter:
                iter += 1
                Y2_d = self.clf.decision_function(X2)  # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d   # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                # print(Y2)
                if positive_id.shape[0] == 0 or negative_id.shape[0] == 0: break
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.vstack([Y1, Y2])
                    self.clf.fit(X3, Y3.ravel(), sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2 * self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu

    def score(self, X, Y):
        return self.clf.score(X, Y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
