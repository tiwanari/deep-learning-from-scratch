import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# rectified linear unit
def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

# def softmax(a):
#     # this implementation causes overflow
#     #exp_a = np.exp(a)
#     #y = exp_a / np.sum(exp_a)
#     #return y
#     c = np.max(a)
#     exp_a = np.exp(a - c) # fix for overflow
#     y = exp_a / np.sum(exp_a)
#     return y

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


# 出力層で利用する活性化関数は
#   回帰問題では恒等関数
#   2クラス分類問題ではsigmoid
#   多クラス問題ではsoftmax
# を使うのが一般的

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    #y = step_function(x)
    y = sigmoid(x)
    #y = relu(x)
    plt.ylim(-0.1,1.1)
    plt.plot(x, y)
    plt.show()
