import numpy as np

# Measure the accuracy of a function (f : A → B) on a dataset of pairs [(A, B)]
def accuracy(f, xs, ys):
    n = len(xs)
    s = 0
    for i in range(0, n):
        yhat = f(xs[i])
        ytrue = ys[i]
        if np.all(yhat == ytrue):
            s += 1
    return s / n
