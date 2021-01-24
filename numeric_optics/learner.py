import numpy as np
from dataclasses import dataclass

from numeric_optics.lens import Lens, identity
from numeric_optics.para import Para

# The Displacement class models a pair of a displacement map and its inverse.
@dataclass
class Displacement:
    displacement: Lens
    inverse: Lens

@dataclass
class Learner:
    model: Para
    update: Lens
    displacement: Displacement

    def to_lens(self):
        displacement = self.displacement.displacement
        inverse_displacement = self.displacement.inverse
        return (self.update @ inverse_displacement) >> self.model.arrow >> displacement

# Vanille gradient descent update lens
def gd(ε):
    """ The vanilla gradient-descent update lens, parametrised by a learning rate ε """
    def update_rev(P):
        p, pdiff = P

        if p is None:
            return None
        elif type(p) is tuple and type(pdiff) is tuple:
            return update_rev((p[0], pdiff[0])), update_rev((p[1], pdiff[1]))
        else:
            return p - ε * pdiff

    return Lens(identity.fwd, update_rev)

# Mean-squared-error displacement map
def mse_rev(args):
    yhat, ytrue = args
    if yhat is None:
        return None
    elif type(yhat) is tuple and type(ytrue) is tuple:
        return mse_rev((yhat[0], ytrue[0])), mse_rev((yhat[1], ytrue[1]))
    else:
        return yhat - ytrue

mse = Displacement(
    displacement=Lens(identity.fwd, mse_rev),
    inverse=Lens(identity.fwd, mse_rev)) # self-inverse
