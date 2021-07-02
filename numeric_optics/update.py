import numpy as np
from dataclasses import dataclass
from typing import Any

import numeric_optics.lens as lens
import numeric_optics.para as para
from numeric_optics.lens import Lens

@dataclass
class Update:
    # S(P) × P → P
    update: Lens

    # P → S(P) -- choose an initial S(P) based on initial params
    initialize: Any = lens.unit_of

# This is like applying a 2-cell in Para, but we also deal with the
# initialization machinery of ParaInit.
def apply_update(para_init: para.ParaInit, update: Update):
    def apply_update_param():
        p0 = para_init.param()
        return update.initialize(p0), p0
    return para.ParaInit(apply_update_param, para.Para((update.update @ lens.identity) >> para_init.arrow.arrow))


def rda_update_fwd(args):
    sp, p = args
    return p

def rda_update_rev(args):
    (sp, p), pdiff = args

    if sp is None and p is None:
        return (None, None)
    elif type(p) is tuple and type(pdiff) is tuple and type(sp) is tuple:
        rv0, rp0 = rda_update_rev(((sp[0], p[0]), pdiff[0]))
        rv1, rp1 = rda_update_rev(((sp[1], p[1]), pdiff[1]))
        return (rv0, rv1), (rp0, rp1)
    else:
        # NOTE: learning rate does not appear here!
        return None, p + pdiff

# The RDA update is just addition of parameters; to recover gradient descent we
# place the learning rate as the "cap" of the learner.
rda = Update(update=Lens(rda_update_fwd, rda_update_rev), initialize=lens.unit_of)

def rda_momentum_update(γ):
    def rda_momentum_rev(args):
        (v, p), pdiff = args

        if v is None and p is None and pdiff is None:
            # Unit object
            return (None, None)
        elif type(v) is tuple and type(p) is tuple and type(pdiff) is tuple:
            # NOTE: because we return TWO results, we have to "unzip" them.
            rv0, rp0 = rda_momentum_rev(((v[0], p[0]), pdiff[0]))
            rv1, rp1 = rda_momentum_rev(((v[1], p[1]), pdiff[1]))
            return (rv0, rv1), (rp0, rp1)
        else:
            # all must be arrays
            vdiff = pdiff + γ * v
            return vdiff, p + vdiff

    return Lens(lens.snd.fwd, rda_momentum_rev)

def rda_momentum(γ):
    return Update(update=rda_momentum_update(γ), initialize=lens.zero_of)

################################################################################
# Old-style updates
# These set the "cap" learning rate as 1, and explicitly multiply by the
# learning rate in the reparametrisation 2-cell
################################################################################

# Vanilla gradient descent update lens
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

    def update_fwd(args):
        sp, p = args
        return p
    return Lens(update_fwd, update_rev)

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

