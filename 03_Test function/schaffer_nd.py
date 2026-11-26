import numpy as np
LB = -100.0
UB = 100.0
D_MIN = 0.5
def f_opt(dim):
    return 0.0
def f(x):
    x = np.asarray(x)
    if len(x) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x) - 1):
        xi, xj = x[i], x[i + 1]
        num = np.sin(xi ** 2 - xj ** 2) ** 2 - 0.5
        den = (1 + 0.001 * (xi ** 2 + xj ** 2)) ** 2
        total += 0.5 + num / den
    return float(total)
