#!/usr/bin/env python

import pathlib
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras import layers, initializers

# The same data as used in non-keras experiments
from experiments.dataset import load_iris

import tensorflow as tf
import pandas as pd
import numpy as np

# Use a fixed learning rate to train
def fixed_learning_rate_scheduler(epoch, lr):
    return lr

_NUM_HIDDEN = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iris-data', default='data/iris.csv')
    parser.add_argument('model', choices=['simple', 'hidden'])
    args = parser.parse_args()

    train_input, train_labels = load_iris(args.iris_data)

    # Build model
    loss = 'mean_squared_error'
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, decay=0)

    # pick a model based on command-line arguments
    if args.model == 'simple':
        model = tf.keras.Sequential([
            Dense(3, input_shape=(4,), activation='sigmoid')
        ])
    elif args.model == 'hidden':
        model = tf.keras.Sequential([
            Dense(_NUM_HIDDEN, input_shape=(4,), activation='sigmoid'),
            Dense(3, activation='sigmoid')
        ])

    # Compile and fit model
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    sched = tf.keras.callbacks.LearningRateScheduler(fixed_learning_rate_scheduler)
    model.fit(train_input, train_labels, epochs=400, batch_size=1, shuffle=True, callbacks=[sched], verbose=2)

    # Print final training accuracy
    score = model.evaluate(train_input, train_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
