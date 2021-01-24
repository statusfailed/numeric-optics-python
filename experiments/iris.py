#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import argparse

from experiments.dataset import load_iris

import numeric_optics.lens as lens
from numeric_optics.para import dense
from numeric_optics.learner import Learner, gd, mse
from numeric_optics.train import train, accuracy

_HIDDEN_LAYER_SIZE = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iris-data', default='data/iris.csv')
    parser.add_argument('model', choices=['simple', 'hidden'])
    args = parser.parse_args()

    # Decide which model to use
    if args.model == "simple":
        # "simple" uses a single dense layer with sigmoid activation (no hidden
        # units)
        model = dense((4, 3), activation=lens.sigmoid)
    if args.model == "hidden":
        # "hidden" uses two dense layers with sigmoid activation and
        # _HIDDEN_LAYER_SIZE hidden units.
        n = _HIDDEN_LAYER_SIZE
        model = dense((4, n), activation=lens.sigmoid) >> dense((n, 3), activation=lens.sigmoid)

    # Load data from CSV
    train_input, train_labels = load_iris(args.iris_data)

    # An extremely simple model with no hidden layer
    learner = Learner(
        model=model,
        update=gd(0.01), # Vanilla gradient descent
        displacement=mse)    # Mean squared error

    e_prev = None
    fwd    = learner.model.arrow.fwd
    for e, j, i, param in train(learner, train_input, train_labels, num_epochs=400, shuffle_data=True):
        # print accuracy diagnostic every epoch
        if e == e_prev:
            continue

        e_prev = e
        f = lambda x: fwd((param, x)).argmax()
        acc = accuracy(f, train_input, train_labels.argmax(axis=1))
        print('epoch', e + 1, '\ttraining accuracy {0:.4f}'.format(acc), end='\r')
    print('epoch', e + 1, '\ttraining accuracy {0:.4f}'.format(acc))
