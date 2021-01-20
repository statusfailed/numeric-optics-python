#!/usr/bin/env python3

import numpy as np
import math
import mnist

import numeric_optics.lens as lens
from numeric_optics.para import Para, dense, relu, sigmoid, to_para, initialize_glorot
from numeric_optics.train import Learner, train, accuracy
import numeric_optics.convolution as image

# Glorot Uniform initialization for 3x3x1 correlation kernel
limit_a = math.sqrt(6 / (3*3 + 1))
a = np.random.uniform(-limit_a, limit_a, ((3,3,3,1)))

# Glorot Uniform initialization for 5x5x3 correlation kernel
limit_b = math.sqrt(6 / (4*4*3 + 1))
b = np.random.uniform(-limit_b, limit_b, ((5,4,4,3)))

model = Para(a, image.multicorrelate) \
     >> relu \
     >> to_para(image.max_pool_3d(2, 2)) \
     >> Para(b, image.multicorrelate) \
     >> relu \
     >> to_para(image.max_pool_3d(2, 2)) \
     >> to_para(image.flatten) \
     >> dense((5*5*5, 10), activation=lens.sigmoid)

if __name__ == "__main__":
    num_classes = 10

    # Load MNIST data
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()

    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255

    # one-hot-encode labels
    y_train = np.identity(num_classes)[y_train]
    y_test  = np.identity(num_classes)[y_test]

    # Reshape images to (28, 28, 1) - 28x28 pixels with a single color channel.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test  = x_test.reshape(x_test.shape + (1,))

    # Train with mean squared error and learning rate 0.01
    learner = Learner(
        model=model,
        update=lens.update(0.01),
        displacement=lens.mse,
        inverse_displacement=lens.identity) # TODO: fix

    # Print diagnostics while training
    e_prev = None
    fwd    = learner.model.arrow.fwd
    for e, j, i, param in train(learner, x_train, y_train, num_epochs=4, shuffle_data=True):
        if j % 10000:
            continue

        e_prev = e
        f = lambda x: fwd((param, x)).argmax()
        # NOTE: this is *TEST* accuracy, unlike iris experiment.
        acc = accuracy(f, x_test, y_test.argmax(axis=1))
        print('epoch', e, 'sample', j, '\taccuracy {0:.4f}'.format(acc), sep='\t')

    # final accuracy
    acc = accuracy(f, x_test, y_test.argmax(axis=1))
    print('final accuracy: {0:.4f}'.format(acc))
