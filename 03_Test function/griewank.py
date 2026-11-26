import numpy as np
LB = -600.0
UB = 600.0
D_MIN = 0.5
def f_opt(dim):
    return 0.0
def f(x):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)
