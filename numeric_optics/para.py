import numpy as np
from numeric_optics import lens

class Para:
    """ Para is the type of *parametrised* maps. """
    def __init__(self, param, arrow):
        self.param = param
        self.arrow = arrow

    def __matmul__(f, g):
        return Para((f.param, g.param), f.arrow @ g.arrow)

    def __rshift__(f, g):
        # NOTE: order of parameters is in reverse with respect to order of composition
        return Para((g.param, f.param), lens.AssocL() >> (lens.Identity() @ f.arrow) >> g.arrow)

def to_para(f):
    """ Lift a Lens into a Para using the unit object as its parameter space """
    return Para(None, lens.Snd() >> f)

################################################################
# Neural network layers
################################################################

# Weight initializers

def initialize_normal(mean, stddev):
    return lambda shape: np.random.normal(mean, stddev, shape)

# Glorot initialization
# http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
def initialize_glorot(shape):
    (b, a) = shape
    stddev = np.sqrt(2.0 / (a + b))
    return np.random.normal(0, stddev, shape)

# A neural network dense layer
# NOTE: this morphism is a composite of the morphisms "linear", "add", and "activation".
def dense(shape, activation, initialize_weights=initialize_normal(0, 0.01)):
    """ Dense neural network layer as a morphism of Para """
    # note: shape is opposite order to matrix dimensions (we write (input, output))
    (a, b) = shape
    p = (np.zeros(b), initialize_weights((b, a)))
    f = lens.AssocL() >> ((lens.Identity() @ lens.Linear()) >> lens.Add() >> activation)
    return Para(p, f)

# Activation layers as zero-parameter morphisms of Para
Sigmoid = to_para(lens.Sigmoid())
ReLU    = to_para(lens.ReLU())
