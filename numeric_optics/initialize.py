""" Initializers are (random) choices of initial parameters """
import numpy as np


def normal(mean=0, stddev=0.01):
    def normal_initializer(shape):
        return np.random.normal(mean, stddev, shape)
    return normal_initializer

# eq (16) http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
def glorot_uniform(shape):
    bound = math.sqrt(6) / math.sqrt(np.product(shape))
    return np.random.uniform

# http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
def glorot_normal(shape):
    (b, a) = shape
    stddev = np.sqrt(2.0 / (a + b))
    return np.random.normal(0, stddev, shape)
