# -*- coding: utf-8 -*-
import numpy as np
import sys, os
sys.path.append(os.pardir)
from ch03.activation_functions import softmax
from ch04.error_functions import cross_entropy_error
from util import im2col, col2im

class Convolution(object):
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W #FN x C x FH x FW (filter num/height/width)
        self.b = b #FN x 1 x 1 (broadcasted to FN x OH x OW)
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad) #(N*OH*OW) x (C*FH*FW (num of convolution block for each dat))
        col_W = self.W.reshape(FN, -1).T               # (C*FH*FW) x FN
        out = np.dot(col, col_W) + self.b              # (N*OH*OW) x FN

        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) #N x FN x OH x OW

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        d_col = np.dot(dout, self.col_W.t)
        dx = col2im(d_col, self.x.shape, FH, FW, self.stride, self.pad)

        self.dW = np.dot(self.col.t, dout).transpose(1, 0).reshape(FN, C, FH, FW)
        self.db = np.sum(dout,axis=0)
        # dx = np.dot(dout, self.w.t)
        # self.dw = np.dot(self.x.t, dout)
        # self.db = np.sum(dout, axis=0)
        return dx

class Pooling(object):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad    = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - 2*self.pad) / self.stride)
        out_w = int(1 + (W - 2*self.pad) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) #(N*OH*OW) x (C*PH*PW)
        col.reshape(-1, self.pool_h*self.pool_w) # (N*OH*OW*C) x (PH*PW)

        out = np.max(col, axis=1)                # (N*OH*OW*C) x 1
        arg_max = np.argmax(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # N x C x OH x OW

        self.arg_max = arg_max
        self.x = x
        return out

    def backward(self, dout):
        # まちがい
        # dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.pool_w*self.pool_h)
        # #d_col = np.zeros((dout.size, self.pool_h*self.pool_w))
        # d_col = dout[self.arg_max]

        # せいかい
        dout = dout.transpose(0, 2, 3, 1) #N x OH x OW x C
        
        pool_size = self.pool_h * self.pool_w
        d_col = np.zeros((dout.size, pool_size))
        d_col[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        #d_col = d_col.reshape(dout.shape + (pool_size,))  # (N*OH*OW*C) x (PH*PW)

        dcol = d_col.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1) # (N*OH*OW) x (C*PH*PW)
        dx = col2im(d_col, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
        

if __name__ == '__main__':

    c = Convolution(0.01*np.random.randn(100, 20, 3, 3), np.zeros((100,)))
    print(c.forward(np.ones((90, 20,10,10))).shape)
    """
    c = Convolution(0.01*np.random.randn(100, 20, 3, 3), np.zeros((100,)))
    c.forward(np.ones((90, 20, 10, 10))).shape
    >>> (90, 100, 8, 8)
    """
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())





