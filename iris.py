#!/usr/bin/env python

import numeric_optics.lens as lens

# TODO: fix imports!
from numeric_optics.lens import *
from numeric_optics.para import *
from numeric_optics.train import *

import pandas as pd

if __name__ == "__main__":
    iris = pd.read_csv('data/iris.csv')

    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)
    # one-hot encode with 3 classes
    train_labels = np.eye(3)[train_labels]


    # An extremely simple model with no hidden layer
    trainable = TrainableModel(
        model=dense((4, 3), activation=lens.sigmoid),
        update=lens.update(0.01), # Vanilla gradient descent
        displacement=lens.mse,    # Mean squared error
        inverse_displacement=lens.identity) # TODO: inverse mean_squared_error map

    e_prev = None
    fwd    = trainable.model.arrow.fwd
    for e, j, i, param in train(trainable, train_input, train_labels, num_epochs=200, shuffle_data=True):
        if e == e_prev:
            continue

        e_prev = e
        f = lambda x: fwd((param, x)).argmax()
        # NOTE: this is *training* accuracy
        acc = accuracy(f, train_input, train_labels.argmax(axis=1))
        print('epoch', e + 1, '\taccuracy {0:.4f}'.format(acc), end='\r')
    print('')
