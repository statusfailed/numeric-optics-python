import pandas as pd
import numpy as np
import mnist

mnist.temporary_dir = lambda: 'data/'
np.random.seed(2)


def load_iris(path):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    return train_input, train_labels


def load_mnist():
    # number of classes
    _MNIST_CLASSES = 10

    # Load MNIST data
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()

    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255

    # one-hot-encode labels
    y_train = np.identity(_MNIST_CLASSES)[y_train]
    y_test  = np.identity(_MNIST_CLASSES)[y_test]

    # Reshape images to (28, 28, 1) -- 28x28 pixels with a single color channel.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test  = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)
