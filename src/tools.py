import numpy as np
import matplotlib.pyplot as plt

config = {
    "2CDT": { "label_size": 5, "drift_examples": 400, "file": "2CDT", "binary": True, "budget": 100},
    "UG_2C_2D": { "label_size": 1, "drift_examples": 1000, "file": "UG_2C_2D", "binary": True, "budget": 100},
}

def gen_arr(p, c): return np.array([c] * p).astype(np.int).reshape(-1, 1)

def swap(x, a, b):
    aix, bix = (x == a), (x == b)
    x[aix], x[bix] = b, a
    return x

def draw(file, json):
    exp_name = json["method"] + '-' + json["dataset"]
    acc = None
    for k in json["acc"]:
        cur = np.array(json["acc"][k])
        acc = cur if acc is None else (acc + cur)
    acc = acc * 100.0 / len(json["acc"])
    x = [i + 1 for i in range(acc.shape[0])]

    plt.clf()
    plt.title(exp_name)
    plt.xlabel("Time Step")
    plt.ylabel("Accuracy (%)")
    plt.plot(x, acc)
    plt.yticks([i * 10 for i in range(11)])
    plt.savefig(file, bbox_inches='tight')