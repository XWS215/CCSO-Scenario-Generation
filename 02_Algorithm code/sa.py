import numpy as np

class SA:
    def __init__(self, dim, bounds, pop_size=50, seed=None):
        self.dim, self.bounds, self.N, self.seed = dim, bounds, pop_size, seed
    def optimize(self, func, max_iter=1000, tol=1e-8, **kw):
        if self.seed is not None: np.random.seed(self.seed)
        lb, ub = self.bounds; D = self.dim; span = ub - lb
        n_iter = max_iter * self.N
        probe = np.random.uniform(lb, ub, (20, D))
        pf = np.array([func(x) for x in probe])
        finite = pf[np.isfinite(pf)]
        T0 = 0.3 * float(np.std(finite)) if len(finite) > 1 else 1.0
        if not np.isfinite(T0) or T0 <= 0: T0 = 1.0
        gi = int(np.argmin(np.where(np.isfinite(pf), pf, np.inf)))
        x, fx = probe[gi].copy(), float(pf[gi])
        best, fbest = x.copy(), fx
        Tend = T0 * 1e-4; step = 0.2 * span; n_acc = acc = 0
        for k in range(n_iter):
            T = T0 * (Tend / T0) ** (k / n_iter)
            y = np.clip(x + step * np.random.randn(D), lb, ub)
            fy = func(y); n_acc += 1
            if fy < fx or np.random.rand() < np.exp(-(fy - fx) / T):
                x, fx = y, float(fy); acc += 1
            if fx < fbest: best, fbest = x.copy(), fx
            if (k + 1) % 100 == 0:
                if acc / n_acc > 0.4: step = np.minimum(step * 1.2, 0.5 * span)
                else: step = np.maximum(step / 1.2, 1e-6 * span)
                n_acc = acc = 0
            if abs(fbest) < tol: break
        return {"best_solution": best, "best_fitness": fbest}
