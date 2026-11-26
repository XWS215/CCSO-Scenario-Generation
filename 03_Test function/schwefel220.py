import numpy as np
LB = -500.0
UB = 500.0
D_MIN = 0.5
def f_opt(dim):
    return -418.9828872724338 * dim
def f(x):
    x = np.asarray(x, dtype=float).ravel()
    return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))
