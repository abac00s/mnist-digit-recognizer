import os
import numpy as np

from functions import xentropy
from layers import LogitLayer, SigmoidLayer


class Network:
    def __init__(self, layers, load_path = None):
        self.layers = layers
        self.load_path = load_path

        if load_path:
            if os.path.isdir(load_path):
                self.load()
            else:
                print('No saved values')

    def load(self):
        for i in range(len(self.layers)):
            self.layers[i].load(os.path.join(self.load_path, str(i)))

    def save(self):
        if not os.path.isdir(self.load_path):
            os.mkdir(self.load_path)

        for i in range(len(self.layers)):
            p = os.path.join(self.load_path, str(i))
            self.layers[i].save(p)

    def forward_prop(self, data):
        caches = []
        a = data
        for layer in self.layers:
            a, cache = layer.forward(a)
            caches.append(cache)
        return a, caches

    def back_prop(self, pred, labels, caches):
        gradients = []

        # it's actually dz
        da = pred - labels

        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            da, dW, db = layer.backward(da, cache)
            gradients.append((dW, db))

        return reversed(gradients)

    def update_weights(self, gradients, lr):
        for layer, gradient in zip(self.layers, gradients):
            layer.update_weights(gradient, lr)

    def accuracy(self, data, labels):
        m = data.shape[1]
        pred, _ = self.forward_prop(data)
        pred = pred.argmax(axis=0)
        labels = labels.argmax(axis=0)
        correct = np.sum(pred == labels)
        return correct / m

    def train(self, data, labels, num_iter, learning_rate, batch_size=100):
        m = data.shape[1]

        batch_num = m // batch_size
        if batch_num * batch_size != m:
            batch_num += 1

        for i in range(num_iter):
            for j in range(batch_num):
                beg, end = j * batch_size, min((j + 1) * batch_size, m)
                batch = data[:, beg:end]
                batch_labels = labels[:, beg:end]

                pred, caches = self.forward_prop(batch)

                gradients = self.back_prop(pred, batch_labels, caches)

                self.update_weights(gradients, learning_rate)

            pred = self.forward_prop(data)[0]
            cost = xentropy(pred, labels)

            acc = self.accuracy(test_images, test_labels)
            print("Cost after {}. iteration = {}, accuracy = {}".format(i, cost, acc))

            if self.load_path:
                self.save()


train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

net = Network([
    SigmoidLayer(28 * 28, 200),
    LogitLayer(200, 10)
], 'w1')

net.train(train_images, train_labels, 1000, 0.1, 100)


