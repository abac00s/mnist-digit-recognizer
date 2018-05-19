import numpy as np

from layers import SigmoidLayer, LogitLayer
from nn import Network

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

net = Network([
    SigmoidLayer(28 * 28, 200),
    LogitLayer(200, 10)
])

net.train(train_images, train_labels, test_images, test_labels, 1000, 0.1, 100)