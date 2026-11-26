import numpy as np
LB = -100.0
UB = 100.0
D_MIN = 0.5
def f_opt(dim):
    return 0.0
def f(x):
    x = np.asarray(x, dtype=float).ravel()
    r = float(np.sqrt(np.sum(x ** 2)))
    return float(1.0 - np.cos(2.0 * np.pi * r) + 0.1 * r)
