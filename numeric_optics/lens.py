from abc import ABC, abstractmethod

import numpy as np
import scipy.special as special

class Lens(ABC):
    """ Monomorphic lenses, stored as a pair of maps
            fwd : A → B
            rev : A × B' → A'
    """
    @abstractmethod
    def fwd(self, x):
        ...

    @abstractmethod
    def rev(self, args):
        ...

    # Tensor
    # f : A -> B
    # g : C -> D
    def __matmul__(f, g):
        """ Tensor product of lenses """
        return Tensor(f, g)

    # Composition
    def __rshift__(f, g):
        """ Composition of lenses in diagrammatic order (f; g) """
        return Compose(f, g)

    def __repr__(self):
        return type(self).__name__

class Compose(Lens):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def fwd(self,x):
        h = self.f.fwd(x)
        return self.g.fwd(h)
        # return self.g.fwd(self.f.fwd(x))
    def rev(self, xy):
        return self.f.rev((  xy[0], self.g.rev(( self.f.fwd(xy[0]), xy[1] )) ))

    def __repr__(self):
        return ('({} >> {})'.format(repr(self.f), repr(self.g)))

class Tensor(Lens):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def fwd(self, ac):
        return (self.f.fwd(ac[0]), self.g.fwd(ac[1]))
    def rev(self, acbd):
        return (self.f.rev((acbd[0][0], acbd[1][0])), self.g.rev((acbd[0][1], acbd[1][1])))

    def __repr__(self):
        return '({} @ {})'.format(repr(self.f), repr(self.g))

################################################################
# Basic lenses

# identity = Lens(lambda x: x, lambda xy: xy[1])
class Identity(Lens):
    def fwd(self, x):
        return x

    def rev(self, xy):
        x, y = xy
        return y

# add = Lens(lambda x: x[0] + x[1], lambda xyz: (xyz[1], xyz[1]))
class Add(Lens):
    def fwd(self, x):
        # NOTE: check x is a tuple, or add along an axis(?)
        return x[0] + x[1]

    def rev(self, xdy):
        x, dy = xdy
        return dy, dy

###############################
# Monoidal and (weak) cartesian maps

class Fst(Lens):
    # Projections for tuples (not arrays!)
    def fwd(self, ab):
        return ab[0]

    def rev(self, args):
        (a, b), da = args
        return da, b

class Snd(Lens):
    def fwd(self, ab):
        return ab[1]

    def rev(self, args):
        (a, b), db = args
        # todo: need a zero that works on tuples too...
        return zero_of(a), db

def assocL_fwd(t):
    ab, c = t
    a, b = ab
    return a, (b, c)

def assocR_fwd(t):
    a, bc = t
    b, c = bc
    return (a, b), c

# assocL = Lens(assocL_fwd, lambda xx: assocR_fwd(xx[1]))
class AssocL(Lens):
    def fwd(self, t):
        return assocL_fwd(t)
    def rev(self, xx):
        return assocR_fwd(xx[1])

# assocR = Lens(assocR_fwd, lambda xx: assocL_fwd(xx[1]))
class AssocR(Lens):
    def fwd(self, t):
        return assocR_fwd(t)
    def rev(self, xx):
        return assocL_fwd(xx[1])

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

# linear = Lens(linear_fwd, linear_rev)
class Linear(Lens):
    def fwd(self, mx):
        """ Forward map of a linear layer """
        m, x = mx
        return m @ x

    def rev(self, mxy):
        """ Reverse map of a linear layer """
        mx, y = mxy
        m, x = mx
        return (np.outer(y, x), m.T @ y)


################################################################
# Activation functions as lenses

# sigmoid = Lens(sigmoid_fwd, sigmoid_rev)
class Sigmoid(Lens):
    def fwd(self, z):
        return special.expit(z)

    def rev(self, xy):
        x, dy = xy
        return self.fwd(x) * (1 - self.fwd(x)) * dy

# relu = Lens(relu_fwd, relu_rev)
class ReLU(Lens):
    def fwd(self, x):
        return np.maximum(x, 0)

    def rev(self, args):
        x, dy = args
        return (x > 0) * dy

###############################
# Update and loss functions

# Lens(identity.fwd, update_rev)
class Update(Lens):
    def __init__(self, ε):
        self.ε = ε

    def fwd(self, x):
        return x

    def rev(self, P):
        p, pdiff = P

        if p is None:
            return None
        elif type(p) is tuple and type(pdiff) is tuple:
            return self.rev((p[0], pdiff[0])), self.rev((p[1], pdiff[1]))
        else:
            return p - self.ε * pdiff

    def __repr__(self):
        return 'Update({})'.format(self.ε)

# mse = Lens(identity.fwd, mse_rev)
class MSE(Lens):
    def fwd(self, x):
        return x

    def rev(self, args):
        yhat, ytrue = args
        if yhat is None:
            return None
        elif type(yhat) is tuple and type(ytrue) is tuple:
            return mse_rev((yhat[0], ytrue[0])), mse_rev((yhat[1], ytrue[1]))
        else:
            return yhat - ytrue
