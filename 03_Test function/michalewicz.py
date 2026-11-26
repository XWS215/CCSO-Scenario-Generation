import numpy as np
LB = 0.0
UB = float(np.pi)
D_MIN = 0.2
F_OPT = {2: -1.8013, 3: -3.1038, 5: -4.6876, 10: -9.6601, 20: -19.63, 30: -29.63}
def f_opt(dim):
    return F_OPT[dim]
def f(x, m=10):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m)))
