import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x)
    s = np.sum(e, axis=0, keepdims=True)
    return e / s


def xentropy(pred, labels):
    m = labels.shape[1]
    return -np.sum(labels * np.log(pred)) / m


def relu(x):
    x = x.copy()
    x[x < 0] = 0
    return x


def relu_derivative(output):
    return output > 0