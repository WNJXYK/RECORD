from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os, h5py
from scipy.io import loadmat
import numpy as np

PATH = "./data"

def load_arff(path):
    with open(path, "r") as fp:
        attrs, data = 0, False
        X, y = [], []
        for line in fp.readlines():
            line = line.strip()
            if len(line) <= 0: continue
            if data:
                X.append([0] * attrs)
                compress = False
                if line.startswith("{"):
                    compress = True
                    line = line[1:-1]
                data = line.split(sep=',')
                if compress:
                    for pair in data:
                        pair = pair.split(sep=' ')
                        if len(pair) < 2: continue
                        X[-1][int(pair[0])] = pair[1]
                else:
                    for i in range(attrs):
                        X[-1][i] = data[i]
            else:
                if line.startswith("@attribute"): attrs += 1
                if line.startswith("@data"): data = True

        if data is False: raise Exception("Dataset: Not a arff format")

        X = np.array(X)
        y = X[:, -1].reshape(-1, 1)
        X = X[:, :-1].reshape(-1, attrs - 1)

        classes = list(set(y[:, 0]))
        rev_classes = {classes[0]: -1, classes[1]: 1}
        for i in range(y.shape[0]):
            y[i, 0] = int(rev_classes[y[i, 0]])

        X = X.astype(np.float)
        y = y.astype(np.int)
        return X, y

def load(name):
    path = os.path.join(PATH, name + ".mat")

    if not os.path.exists(path):
        raise Exception("Dataset: File not found.")

    X, y = None, None
    # H5py
    if X is None:
        try:
            data = h5py.File(path, mode="r")
            X, y = np.transpose(data["x"]), np.transpose(data["y"]).astype(np.int)
        except:
            pass
    # Scipy_x_y
    if X is None:
        try:
            data = loadmat(path)
            X, y = np.array(data["x"]), np.array(data["y"]).astype(np.int)
        except:
            pass
    # Scipy_data_label
    if X is None:
        try:
            data = loadmat(path)
            X, y = np.transpose(data["data"]), np.transpose(data["label"]).astype(np.int)
        except:
            pass
    # Arff
    if X is None:
        try:
            X, y = load_arff(path)
        except:
            pass

    if X is None:
        raise Exception("Dataset: Format not supported.")

    return X, y

class Dataset:
    def get_class_examples(self, X, y, c, m):
        ix = y[:, 0] == c
        X, y = X[ix, :], y[ix, :]
        n = X.shape[0]
        ix, perm = None, np.arange(n)
        if m <= n: ix = np.random.choice(perm, m, replace=False)
        if m > n: ix = np.random.choice(perm, m, replace=True)

        return X[ix], y[ix]

    def __init__(self, dataset="1CHT", labeled_siz=20, budget=100, random_seed=19260817, drift_examples=300):
        # Read Data
        X, y = load(dataset)
        self.X, self.y = X, y

        self.n_features, n = X.shape[1], X.shape[0]
        self.classes = np.array(list(set(y[:, 0]) - set([-7])))
        self.n_classes = self.classes.shape[0]

        self.drift_examples = drift_examples
        self.budget = budget
        self.labeled_size = labeled_siz
        self.n_batches, self.cur_batch = int(n / self.drift_examples) - 1, 0

        # Init Environment Storage
        self.saved_X, self.saved_y = np.zeros((0, self.n_features)), np.zeros((0, 1))
        self.saved_unlabeled = np.zeros((0, self.n_features))
        self.quiz, self.answ = np.zeros((0, self.n_features)), np.zeros(0)

        # Set Random Seed
        self.random_seed = random_seed
        if self.random_seed is not None: np.random.seed(self.random_seed)

        # Get Labeled Examples
        for c in self.classes:
            cur_X, cur_y = self.get_class_examples(self.X[:self.drift_examples, :], self.y[:self.drift_examples, :], c, labeled_siz)
            self.saved_X = np.vstack([self.saved_X, cur_X])
            self.saved_y = np.vstack([self.saved_y, cur_y])
        self.X, self.y = self.X[self.drift_examples:, :], self.y[self.drift_examples:, :]

    def get_labeled_X(self): return self.saved_X
    def get_labeled_y(self): return self.saved_y
    def get_unlabeled(self): return self.saved_unlabeled

    def next(self):
        if self.random_seed is not None: np.random.seed(self.random_seed)

        self.cur_batch += 1
        X, y = self.X[:self.drift_examples, :], self.y[:self.drift_examples, :]
        self.X, self.y = self.X[self.drift_examples:, :], self.y[self.drift_examples:, :]
        unlabeled_X, self.quiz, unlabeled_y, self.answ = train_test_split(X, y, test_size=0.3)
        return unlabeled_X, unlabeled_y

    def evaluate(self, pred_y):
        return accuracy_score(y_true=self.answ.ravel(), y_pred=pred_y)