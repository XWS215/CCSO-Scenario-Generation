import numpy as np
LB = -100.0
UB = 100.0
D_MIN = 0.5
def f_opt(dim):
    return 0.0
def f(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return float(x[0] ** 2)
    return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))
