# -*- coding: utf-8 -*-
import numpy as np
import sys, os
sys.path.append(os.pardir)
from ch03.activation_functions import softmax
from ch04.error_functions import cross_entropy_error


"""
just a partial differential problem...
that's too a redundant explanation.
"""

class Relu(object):
    def __init__(self, ):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid(object):
    def __init__(self, ):
        self.out = None

    def forward(self, x):
        self.out = 1/ (1+np.exp(-x))
        return self.out
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss(object):
    # softmax + cross entropy error
    def __init__(self, ):
        self.loss = None
        self.y = None
        self.t = None #one-hot vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx




