import numpy as np
from dataclasses import dataclass
from typing import Any

import numeric_optics.lens as lens
from numeric_optics.lens import Lens, identity
from numeric_optics.para import Para

# The Displacement class models a pair of a displacement map and its inverse.
@dataclass
class Displacement:
    displacement: Lens
    inverse: Lens

@dataclass
class Update:
    # S(P) × P → P
    update: Lens

    # P → S(P) -- choose an initial S(P) based on initial params
    initialize: Any = lens.unit_of

@dataclass
class Learner:
    model: Para
    update: Update
    displacement: Displacement

    def to_lens(self):
        displacement = self.displacement.displacement
        inverse_displacement = self.displacement.inverse
        update = self.update.update
        return (update @ inverse_displacement) >> self.model.arrow >> displacement

# Vanille gradient descent update lens
def gd_update(ε):
    """ The vanilla gradient-descent update lens, parametrised by a learning rate ε """
    # args : S(P) × P
    # with   S(P) = I
    # returns: S(P) × P
    def update_rev(args):
        (sp, p), pdiff = args

        if sp is None and p is None:
            return (None, None)
        elif type(p) is tuple and type(pdiff) is tuple and type(sp) is tuple:
            rv0, rp0 = update_rev(((sp[0], p[0]), pdiff[0]))
            rv1, rp1 = update_rev(((sp[1], p[1]), pdiff[1]))
            return (rv0, rv1), (rp0, rp1)
        else:
            return None, p - ε * pdiff

    return Lens(lens.snd.fwd, update_rev)

def gd(ε):
    return Update(update=gd_update(ε), initialize=lens.unit_of)

def momentum_update(ε, γ):
    """ Momentum gradient descent, with learning rate ε and momentum γ """
    def momentum_rev(args):
        (v, p), pdiff = args

        if v is None and p is None and pdiff is None:
            # Unit object
            return (None, None)
        elif type(v) is tuple and type(p) is tuple and type(pdiff) is tuple:
            # NOTE: because we return TWO results, we have to "unzip" them.
            rv0, rp0 = momentum_rev(((v[0], p[0]), pdiff[0]))
            rv1, rp1 = momentum_rev(((v[1], p[1]), pdiff[1]))
            return (rv0, rv1), (rp0, rp1)
        else:
            # all must be arrays
            vdiff = γ * v + ε * pdiff
            return vdiff, p - vdiff

    return Lens(lens.snd.fwd, momentum_rev)

def momentum(ε, γ):
    return Update(update=momentum_update(ε, γ), initialize=lens.zero_of)


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
