#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import argparse

from experiments.dataset import load_iris

import numeric_optics.lens as lens
from numeric_optics.para import Para, to_para, dense, linear
from numeric_optics.supervised import train_supervised, supervised_step, supervised_step_para, mse_loss, learning_rate
from numeric_optics.update import gd, rda
from numeric_optics.statistics import accuracy

_HIDDEN_LAYER_SIZE = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iris-data', default='data/iris.csv')
    parser.add_argument('model', choices=['linear', 'dense', 'hidden'])
    args = parser.parse_args()

    # Decide which model to use
    if args.model == "linear":
        model = linear((4, 3))
    if args.model == "dense":
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

    # Create the "step" function P × A × B → P
    step, param = supervised_step_para(model, rda, Para(mse_loss), to_para(learning_rate(η=-0.01)))

    e_prev = None
    fwd = model.arrow.arrow.fwd
    for e, j, i, param in train_supervised(step, param, train_input, train_labels, num_epochs=400, shuffle_data=True):
        # print accuracy diagnostic every epoch
        if e == e_prev:
            continue

        e_prev = e
        predict = lambda x: fwd((param[1], x)).argmax()
        acc = accuracy(predict, train_input, train_labels.argmax(axis=1))
        print('epoch', e + 1, '\ttraining accuracy {0:.4f}'.format(acc), end='\r')
    print('epoch', e + 1, '\ttraining accuracy {0:.4f}'.format(acc))
