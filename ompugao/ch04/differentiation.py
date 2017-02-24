import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        x_clone = x
        x_clone[idx] = x[idx] + h
        fxh1 = f(x_clone)
        x_clone[idx] = x[idx] - h
        fxh2 = f(x_clone)
        grad[idx] = (fxh1 - fxh2) / (2*h)
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    lr: learning rate
    """
    x = init_x

    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)
    return x


if __name__ == '__main__':
    def func(x):
        return x[0] ** 2 + x[1] ** 2
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(func, init_x=init_x, lr=0.1, step_num=100))

