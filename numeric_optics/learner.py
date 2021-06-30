import numpy as np
from dataclasses import dataclass
from typing import Any

import numeric_optics.lens as lens
from numeric_optics.lens import Lens, identity
from numeric_optics.para import Para

from numeric_optics.update import *

# The Displacement class models a pair of a displacement map and its inverse.
@dataclass
class Displacement:
    # TODO: split displacement into displacement + inverse_displacement
    # make inverse_displacement optional
    displacement: Lens
    inverse: Lens

@dataclass
class Learner:
    model: Para
    update: Update
    displacement: Displacement

    def to_lens(self):
        displacement = self.displacement.displacement
        inverse_displacement = self.displacement.inverse
        update = self.update.update
        # NOTE: this is the same as applying a 2-cell
        return (update @ inverse_displacement) >> self.model.arrow >> displacement

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
