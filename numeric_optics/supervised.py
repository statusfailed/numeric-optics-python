import numpy as np
from dataclasses import dataclass
from typing import Any

import numeric_optics.lens as lens
from numeric_optics.lens import Lens, identity
from numeric_optics.para import Para, ParaInit
from numeric_optics.update import Update, apply_update

# Loss : (B × B → L, B × B × L' → B' × B')
# LR : (L → I, L × I' → L')
# LR = (lambda loss: None, lambda loss, _: η)
def learning_rate(η: float):
    # note: rev creates constant array of η based on dimensions of loss
    def learning_rate_fwd(loss):
        return None

    def learning_rate_rev(args):
        loss, unit = args
        assert unit is None
        return np.array([η])

    return Lens(learning_rate_fwd, learning_rate_rev)

# TODO: FIXME!
def mse_fwd(args):
    y, yhat = args
    return np.array([0]) # TODO: FIXME!

def mse_rev(args):
    # note: 'loss' represents a *change* in loss, and is normally a constant-
    # the learning rate η
    (y, yhat), loss = args
    assert type(loss) is np.ndarray
    return loss * (y - yhat), loss * (yhat - y)

mse_loss = Lens(mse_fwd, mse_rev)

# Returns a function of type P × A × B
def supervised_step(model: ParaInit, update: Update, loss: Lens, learning_rate: Lens):
    model_arrow = model.arrow.arrow # ParaInit → Para → Lens

    # Create step function
    # (B × (S(P) × P)) × A
    morphism = ((lens.identity @ update.update) @ lens.identity) >> lens.assocL >> (lens.identity @ model_arrow) >> loss >> learning_rate

    def step(b, p, a):
        # NOTE: it's important we pass "None" as an input here: this is because
        # the type of the reverse map is P × A × B × I → P × A × B,
        # the "None" is corresponds to the type I.
        (b_new, p_new), a_new = morphism.rev((((b, p), a), None))
        return p_new

    # Create initial parameters
    p0 = model.param()            # initialize model parameters P
    param = update.initialize(p0) # initialize S(P) using P

    return step, (param, p0)
    # return morphism, (param, p0)

# TODO?
# def supervised_step_para(model: Para, update: Update, loss: Para, cap: Para):
    # bend >> apply_update(update, model) >> loss >> cap

# step : (S(P) × P) × A × B → (S(P) × P)
# initial_parameters : S(P) × P
def train_supervised(step, initial_parameters, train_x, train_y, num_epochs=1, shuffle_data=True):
    # Check we have the same number of features and labels
    n = np.shape(train_x)[0]
    m = np.shape(train_y)[0]
    if n != m:
        err = "Mismatch in dimension 0: {} training examples but {} labels".format(n, m)
        raise ValueError(err)

    xs    = train_x
    ys    = train_y
    permutation = np.array(range(0, n))
    param = initial_parameters
    for epoch in range(0, num_epochs):
        if shuffle_data:
            np.random.shuffle(permutation)

        # A single loop of "generalised SGD" over each training example
        for j in range(0, n):
            i = permutation[j] # for shuffling
            x, y = xs[i], ys[i]
            param = step(y, param, x)
            yield (epoch, j, i, param)

