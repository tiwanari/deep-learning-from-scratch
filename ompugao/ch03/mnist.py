import sys
import os

sys.path.append(os.path.join(os.pardir, os.pardir))
from dataset.mnist import load_mnist
from activation_functions import *
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
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
    y = softmax(a3)

    return y
if __name__ == '__main__':
    #(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    #print(x_train.shape)
    #print(t_train.shape)
    #print(x_test.shape)
    #print(t_test.shape)
    accuracy_count = 0
    network = init_network()
    x, t = get_data()
    batch_size = 100

    if batch_size is None:
        for i in range(len(x)):
            y = predict(network,x[i])
            p = np.argmax(y) #use the most probable index
            if p == t[i]:
                accuracy_count+=1
    else:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_count += np.sum(p == t[i:i+batch_size])
    print("accuracy: ", str(float(accuracy_count/len(x))))
