#!/usr/bin/env python

import sys
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers

from experiments.dataset import load_mnist

_NUM_CLASSES = 10
_INPUT_SHAPE = (28, 28, 1)
_NUM_EPOCHS  = 4

def fixed_learning_rate_scheduler(epoch, lr):
    return lr

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = keras.Sequential([
        keras.Input(shape=_INPUT_SHAPE),
        layers.Conv2D(3, kernel_size=(3, 3), activation="relu", use_bias=False),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(5, kernel_size=(4, 4), activation="relu", use_bias=False),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(_NUM_CLASSES, activation="sigmoid"), # TODO: change me to softmax TODO
    ])

    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, decay=0)
    sched = tf.keras.callbacks.LearningRateScheduler(fixed_learning_rate_scheduler)

    model.summary()

    # Train model
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=_NUM_EPOCHS, batch_size=1, shuffle=True, callbacks=[sched])

    # Evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
