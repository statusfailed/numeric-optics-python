import numpy as np
from numeric_optics import lens
from numeric_optics import initialize

class Para:
    """ Parametrised maps """
    def __init__(self, arrow):
        assert type(arrow) is lens.Lens
        self.arrow = arrow

    def __matmul__(f, g):
        assert False
        assert type(f) is Para
        assert type(g) is Para
        # FIXME: add ex
        return Para(f.arrow @ g.arrow)

    def __rshift__(f, g):
        # NOTE: order of parameters is in reverse with respect to order of composition
        assert type(f) is Para
        assert type(g) is Para
        return Para(lens.assocL >> (lens.identity @ f.arrow) >> g.arrow)

class ParaInit:
    """ Parametrised maps plus an "initializer" for the parameters """
    def __init__(self, param, arrow: Para):
        assert type(arrow) is Para
        self.param = param
        self.arrow = arrow

    def __matmul__(f, g):
        return ParaInit(lambda: (f.param(), g.param()), f.arrow @ g.arrow)

    def __rshift__(f, g):
        assert type(f) is ParaInit
        assert type(g) is ParaInit
        # NOTE: order of parameters is in reverse with respect to order of composition
        return ParaInit(lambda: (g.param(), f.param()), f.arrow >> g.arrow)

def to_para(f):
    """ Lift a Lens into a Para using the unit object as its parameter space """
    return Para(lens.snd >> f)

def to_para_init(f):
    """ Lift a Lens into a ParaInit """
    return ParaInit(lambda: None, to_para(f))

################################################################
# Neural network layers
################################################################

# Activation layers as zero-parameter morphisms of Para
sigmoid = to_para_init(lens.sigmoid)
relu    = to_para_init(lens.relu)

def linear(shape, initialize=initialize.normal(0, 0.01)):
    a, b = shape
    p = lambda: initialize((b, a))
    return ParaInit(p, Para(lens.linear))

def bias(n: int, initialize=np.zeros):
    return ParaInit(lambda: initialize((n,)), Para(lens.add))

# A neural network dense layer
# NOTE: this morphism is a composite of the morphisms "linear", "add", and "activation".
def dense(shape: tuple, activation: lens.Lens):
    return linear(shape) >> bias(shape[1]) >> to_para_init(activation)
