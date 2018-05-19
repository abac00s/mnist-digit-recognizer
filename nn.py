import os
import numpy as np

from functions import xentropy


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

    def backprop(self, pred, labels, caches):
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

    def train(self, examples, labels, test_examples, test_labels, num_iter, learning_rate, batch_size=100):
        m = examples.shape[1]
        num_labels = labels.shape[0]

        batch_num = m // batch_size
        if batch_num * batch_size != m:
            batch_num += 1

        data = np.vstack((labels, examples)).T

        for i in range(num_iter):
            np.random.shuffle(data)

            for j in range(batch_num):
                beg, end = j * batch_size, min((j + 1) * batch_size, m)
                batch_examples = data[beg:end, num_labels:].T
                batch_labels = data[beg:end, :num_labels].T

                pred, caches = self.forward_prop(batch_examples)

                gradients = self.backprop(pred, batch_labels, caches)

                self.update_weights(gradients, learning_rate)

            pred = self.forward_prop(examples)[0]
            cost = xentropy(pred, labels)

            acc = self.accuracy(test_examples, test_labels)
            print("Cost after {}. iteration = {}, accuracy = {}".format(i, cost, acc))

            if self.load_path:
                self.save()



