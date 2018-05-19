import numpy as np
from struct import pack, unpack


def import_labels(data):
    magic, count = unpack('>II', data[:8])
    assert(magic == 2049)
    return list(data[8:])


def import_images(data):
    magic, count, rows, columns = unpack('>IIII', data[:16])
    assert(magic == 2051)
    assert(rows == 28 and columns == 28)
    n = rows*columns
    result = np.zeros((count, n), dtype=np.float32)
    for i in range(count):
        for j in range(n):
            result[i,j] = data[i*n + j + 16]
    return result


def one_hot(labels, ndim):
    m = len(labels)
    res = np.zeros((ndim, m), dtype=np.float32)
    for i in range(m):
        res[labels[i], i] = 1
    return res


with open('train-images-idx3-ubyte', 'rb') as f:
    train_images = import_images(f.read())

with open('train-labels-idx1-ubyte', 'rb') as f:
    train_labels = import_labels(f.read())

with open('t10k-images-idx3-ubyte', 'rb') as f:
    test_images = import_images(f.read())

with open('t10k-labels-idx1-ubyte', 'rb') as f:
    test_labels = import_labels(f.read())

train_images = train_images.T
test_images = test_images.T
train_labels = one_hot(train_labels, 10)
test_labels = one_hot(test_labels, 10)

np.save('train_images', train_images)
np.save('test_images', test_images)
np.save('train_labels', train_labels)
np.save('test_labels', test_labels)
