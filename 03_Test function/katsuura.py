import numpy as np
LB = -100.0
UB = 100.0
D_MIN = 0.5
def f_opt(dim):
    return 0.0
def f(x):
    x = np.asarray(x, dtype=float).ravel()
    d = x.size
    j = 2.0 ** np.arange(1, 33, dtype=float)
    z = j[None, :] * x[:, None]
    inner = np.sum(np.abs(z - np.round(z)) / j[None, :], axis=1)
    p = np.prod((1.0 + np.arange(1, d + 1, dtype=float) * inner) ** (10.0 / d ** 1.2))
    return float((10.0 / d ** 2) * p - 10.0 / d ** 2)
