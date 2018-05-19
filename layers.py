import numpy as np
import os

from functions import relu, relu_derivative, softmax, sigmoid


class Layer:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_out, n_in) * 0.1
        self.b = np.random.randn(n_out, 1) * 0.1

    def update_weights(self, gradients, lr):
        dW, db = gradients
        self.W = self.W - lr * dW
        self.b = self.b - lr * db

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        np.save(os.path.join(path, 'weights.npy'), self.W)
        np.save(os.path.join(path, 'biases.npy'), self.b)

    def load(self, path):
        self.W = np.load(os.path.join(path, 'weights.npy'))
        self.b = np.load(os.path.join(path, 'biases.npy'))


class ReluLayer(Layer):
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_out, n_in) * 0.001
        self.b = np.random.randn(n_out, 1) * 0.001

    def forward(self, data):
        res = relu(self.W.dot(data) + self.b)
        cache = data, res
        return res, cache

    def backward(self, da, cache):
        m = da.shape[1]
        a_prev, a = cache
        dz = relu_derivative(a)
        dW = dz.dot(a_prev.T) / m
        db = dz.sum(axis=1, keepdims=True) / m
        da_prev = self.W.T.dot(dz)
        return da_prev, dW, db


class LogitLayer(Layer):
    def forward(self, data):
        cache = data
        return softmax(self.W.dot(data) + self.b), cache

    def backward(self, dz, cache):
        m = dz.shape[1]
        a_prev = cache
        dW = dz.dot(a_prev.T) / m
        db = dz.sum(axis=1, keepdims=True) / m
        da_prev = self.W.T.dot(dz)
        return da_prev, dW, db


class SigmoidLayer(Layer):
    def forward(self, data):
        res = sigmoid(self.W.dot(data) + self.b)
        cache = data, res
        return res, cache

    def backward(self, da, cache):
        m = da.shape[1]
        a_prev, a = cache
        dz = da * a * (1 - a)
        dW = dz.dot(a_prev.T) / m
        db = dz.sum(axis=1, keepdims=True) / m
        da_prev = self.W.T.dot(dz)
        return da_prev, dW, db