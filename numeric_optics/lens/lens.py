import numpy as np
import scipy.special as special

class Lens:
    """ Monomorphic lenses, stored as a pair of maps
            fwd : A → B
            rev : A × B' → A'
    """
    def __init__(self, fwd, rev):
        self.fwd = fwd
        self.rev = rev

    # Tensor
    # f : A -> B
    # g : C -> D
    def __matmul__(f, g):
        """ Tensor product of lenses """
        fwd = lambda ac: (f.fwd(ac[0]), g.fwd(ac[1]))
        rev = lambda acbd: (f.rev((acbd[0][0], acbd[1][0])), g.rev((acbd[0][1], acbd[1][1])))
        return Lens(fwd, rev)

    # Composition
    def __rshift__(f, g):
        """ Composition of lenses in diagrammatic order (f; g) """
        fwd = lambda x: g.fwd(f.fwd(x))
        rev = lambda xy: f.rev((  xy[0], g.rev(( f.fwd(xy[0]), xy[1] )) ))
        return Lens(fwd, rev)

################################################################
# Basic lenses

identity = Lens(lambda x: x, lambda xy: xy[1])
add = Lens(lambda x: x[0] + x[1], lambda xyz: (xyz[1], xyz[1]))

###############################
# Monoidal and (weak) cartesian maps

# Projections for tuples (not arrays!)
def fst_fwd(ab):
    return ab[0]

def fst_rev(args):
    (a, b), da = args
    return da, np.zeros(b.shape)

fst = Lens(fst_fwd, fst_rev)

def snd_fwd(ab):
    return ab[1]

def snd_rev(args):
    (a, b), db = args
    # todo: need a zero that works on tuples too...
    return zero_of(a), db

snd = Lens(snd_fwd, snd_rev)

def assocL_fwd(t):
    ab, c = t
    a, b = ab
    return a, (b, c)

def assocR_fwd(t):
    a, bc = t
    b, c = bc
    return (a, b), c

assocL = Lens(assocL_fwd, lambda xx: assocR_fwd(xx[1]))
assocR = Lens(assocR_fwd, lambda xx: assocL_fwd(xx[1]))

def unit_of(x):
    """ Map a point A to the corresponding unit object """
    if x is None:
        return None
    elif type(x) is tuple:
        return unit_of(x[0]), unit_of(x[1])
    else:
        return None

def zero_of(x):
    """ Map a point A to the zero map of its type of changes (0 : I → A') """
    if x is None:
        return None
    elif type(x) is tuple:
        return zero_of(x[0]), zero_of(x[1])
    else:
        return np.zeros(x.shape)


################################################################
# Neural Network layers and activation functions

def linear_fwd(mx):
    """ Forward map of a linear layer """
    m, x = mx
    return m @ x

def linear_rev(mxy):
    """ Reverse map of a linear layer """
    (m, x), y = mxy
    return (np.outer(y, x), m.T @ y)

linear = Lens(linear_fwd, linear_rev)

################################################################
# Activation functions as lenses

def sigmoid_fwd(z):
    return special.expit(z)

def sigmoid_rev(xy):
    x, dy = xy
    return sigmoid_fwd(x) * (1 - sigmoid_fwd(x)) * dy

def relu_fwd(x):
    return np.maximum(x, 0)

def relu_rev(args):
    x, dy = args
    return (x > 0) * dy

sigmoid = Lens(sigmoid_fwd, sigmoid_rev)
relu = Lens(relu_fwd, relu_rev)
