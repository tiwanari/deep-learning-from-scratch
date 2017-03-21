#!/usr/bin/env python3
import pickle
import sys
import numpy as np
from sigmoid import sigmoid
from softmax import softmax

sys.path.append('../..')
from dataset.mnist import load_mnist


def get_date():
    (x_train, l_train), (x_test, l_test) \
        = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, l_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    return softmax(a3)

x, l = get_date()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == l[i]:
        accuracy_cnt += 1

print('Accuracy:' + str(float(accuracy_cnt) / len(x)))
