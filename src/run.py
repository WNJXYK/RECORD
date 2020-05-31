import sys, os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse, json, time
from src.RECORD import RECORDS_choice
from src.dataset import Dataset
import numpy as np
from src.methods.label_propagation import LabelPropagation
from src.methods.S3VM import S3VM
from src.methods.DSSL import MT
from tqdm import trange
from src.tools import gen_arr, swap, draw, config

parser = argparse.ArgumentParser()
parser.add_argument('--method', '-m', default="LP", choices=["LP", "S3VM", "DSSL"])
parser.add_argument('--seed', '-s', default=0, type=int)
parser.add_argument('--repeat', '-r', default=5, type=int)
parser.add_argument('--dataset', '-d', default='2CDT', choices=["2CDT", "UG_2C_2D"])
args = parser.parse_args()


def run(seed=0):
    env = Dataset(config[args.dataset]["file"],
                  labeled_siz=config[args.dataset]["label_size"],
                  budget=config[args.dataset]["budget"],
                  random_seed=seed,
                  drift_examples=config[args.dataset]["drift_examples"])
    binary = config[args.dataset]["binary"]
    if not binary and args.method == "S3VM": raise Exception("S3VM only for 2-class data sets.")

    arr = []
    for T in trange(env.n_batches):
        # Get Unlabeled data
        unlabeled_X, unlabeled_y = env.next()
        selected_unlabeled = env.get_unlabeled()
        selected_unlabeled = np.vstack([env.get_unlabeled(), unlabeled_X])

        # Train
        X = np.vstack([env.get_labeled_X(), selected_unlabeled])
        clf = None
        if args.method == "LP":
            if binary:
                y = np.vstack([env.get_labeled_y(), gen_arr(selected_unlabeled.shape[0], 0)]).ravel()
                y = swap(y, 0, -1)
            else:
                y = np.vstack([env.get_labeled_y(), gen_arr(selected_unlabeled.shape[0], -1)]).ravel()
            clf = LabelPropagation(kernel="rbf", n_neighbors=9, max_iter=100000, n_jobs=-1)
            clf.fit(X, y)
        if args.method == "S3VM":
            clf = S3VM(max_iter=300, kernel="rbf")
            clf.fit(env.get_labeled_X(), env.get_labeled_y().ravel(), selected_unlabeled)
        if args.method == "DSSL":
            clf = MT(env.classes, env.n_features)
            clf.fit(env.get_labeled_X(), selected_unlabeled, env.get_labeled_y().ravel())

        # Test
        if args.method == "LP":
            pred_y = clf.predict(env.quiz)
            if binary: pred_y = swap(pred_y, 0, -1)
        if args.method == "S3VM":
            pred_y = clf.predict(env.quiz).reshape(-1, 1)
        if args.method == "DSSL":
            pred_y = clf.predict(env.quiz)
        acc = env.evaluate(pred_y)
        arr.append(acc)

        # Select
        pred, proba = clf.predict(selected_unlabeled), clf.predict_proba(selected_unlabeled)
        if args.method == "LP" and binary: pred = swap(pred, 0, -1)
        if args.method == "S3VM" or args.method == "DSSL": pass
        env.saved_X, env.saved_y, env.saved_unlabeled = RECORDS_choice(env.saved_X, env.saved_y, selected_unlabeled,
                                                                       pred, proba, env.budget,
                                                                       m_last=unlabeled_X.shape[0])

    return arr


if __name__ == "__main__":
    exp_name = "{0}-{1}".format(args.method, args.dataset)
    print(exp_name)

    logs = {"acc": {}, "method": args.method, "dataset": args.dataset, "seed": args.seed}
    for run_index in range(args.repeat):
        arr = run(args.seed + run_index)
        logs["acc"][str(args.seed + run_index)] = arr

    for k in logs["acc"]:
        print(k, logs["acc"][k])

    res_filename = exp_name + '-' + str(time.time())
    with open("./logs/" + res_filename + ".json", 'w') as logfile:
        json.dump(logs, logfile)
    draw("./images/" + res_filename + ".png", logs)



