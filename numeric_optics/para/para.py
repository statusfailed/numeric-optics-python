import numpy as np
from numeric_optics import lens
from numeric_optics import initialize

class Para:
    """ Para is the type of *parametrised* maps. """
    def __init__(self, param, arrow):
        self.param = param
        self.arrow = arrow

    def __matmul__(f, g):
        return Para((f.param, g.param), f.arrow @ g.arrow)

    def __rshift__(f, g):
        # NOTE: order of parameters is in reverse with respect to order of composition
        return Para((g.param, f.param), lens.assocL >> (lens.identity @ f.arrow) >> g.arrow)

def to_para(f):
    """ Lift a Lens into a Para using the unit object as its parameter space """
    return Para(None, lens.snd >> f)

################################################################
# Neural network layers
################################################################

# Weight initializers

# A neural network dense layer
# NOTE: this morphism is a composite of the morphisms "linear", "add", and "activation".
def dense(shape, activation, initialize_weights=initialize.normal(0, 0.01)):
    """ Dense neural network layer as a morphism of Para """
    # note: shape is opposite order to matrix dimensions (we write (input, output))
    (a, b) = shape
    p = (np.zeros(b), initialize_weights((b, a)))
    f = lens.assocL >> (lens.identity @ lens.linear) >> lens.add >> activation
    return Para(p, f)

# Activation layers as zero-parameter morphisms of Para
sigmoid = to_para(lens.sigmoid)
relu    = to_para(lens.relu)
