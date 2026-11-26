import numpy as np
LB = -5.12
UB = 5.12
D_MIN = 0.5
def f_opt(dim):
    return 0.0
def f(x):
    x = np.asarray(x)
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))
