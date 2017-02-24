import numpy as np
import sys, os
sys.path.append(os.path.join(os.pardir, os.pardir))
from dataset.mnist import load_mnist

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t, batch=True, onehot=True):
    if not batch:
        delta = 1e-7
        return -np.sum(t*np.log(y+delta))
    else:
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        if onehot:
            return -np.sum(t*np.log(y)) / batch_size
        else:
            #label data (like "2", "7" etc.)
            return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size




