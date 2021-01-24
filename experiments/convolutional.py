#!/usr/bin/env python3

import numpy as np
import math

from experiments.dataset import load_mnist

import numeric_optics.lens as lens
from numeric_optics.para import Para, dense, relu, sigmoid, to_para
from numeric_optics.train import Learner, train, accuracy
# import numeric_optics.lens.convolution as image
import numeric_optics.para.convolution as image

model = image.multicorrelate((3,3,3,1)) \
     >> relu \
     >> image.max_pool_3d(2, 2) \
     >> image.multicorrelate((5,4,4,3)) \
     >> relu \
     >> image.max_pool_3d(2, 2) \
     >> image.flatten \
     >> dense((5*5*5, 10), activation=lens.sigmoid)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

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
        # only print diagnostics every 10Kth sample
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
