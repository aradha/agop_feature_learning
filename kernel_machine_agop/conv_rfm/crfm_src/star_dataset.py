import torch
import numpy as np
import random

SEED = 1717
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 32


def draw_star(ex, v, offset=(0, 0), c=3):
    q, r = offset
    ex[-1, 5+q:6+q, 7+r:14+r] = v
    ex[-1, 4+q, 9+r:12+r] = v
    ex[-1, 3+q, 10+r] = v
    ex[-1, 6+q, 8+r:13+r] = v
    ex[-1, 7+q, 9+r:12+r] = v
    ex[-1, 8+q, 8+r:13+r] = v
    ex[-1, 9+q, 8+r:10+r] = v
    ex[-1, 9+q, 11+r:13+r] = v
    return ex

def star_dataset(num_samples=200, size=(3, IMG_SIZE, IMG_SIZE)):
    labelset = {}
    for i in range(2):
        one_hot = torch.zeros(2)
        one_hot[i] = 1
        labelset[i] = one_hot

    X = []
    Y = []

    for i in range(num_samples):
        q = random.randint(0, IMG_SIZE - 15)
        r = random.randint(0, IMG_SIZE - 15)
        ex = np.random.normal(size=size) * 3e-1
        ex = torch.from_numpy(ex).float()
        ex = draw_star(ex, 1, offset=(q, r), c=3)

        y = 1
        y_val = labelset[y]
        X.append(ex)
        Y.append(y_val)

    for i in range(num_samples):
        ex = np.random.normal(size=(ex.shape)) * 3e-1
        ex = torch.from_numpy(ex).float()
        y = 0
        y_val = labelset[y]
        X.append(ex)
        Y.append(y_val)
     
    X, y = torch.from_numpy(np.stack(X)).float(), torch.from_numpy(np.stack(Y)).float()

    idx = torch.randperm(len(X))
    return X[idx], y[idx]
