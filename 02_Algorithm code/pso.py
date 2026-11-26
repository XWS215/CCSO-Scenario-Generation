import numpy as np

class PSO:
    def __init__(self, dim, bounds, pop_size=50, seed=None):
        self.dim, self.bounds, self.N, self.seed = dim, bounds, pop_size, seed
    def optimize(self, func, max_iter=1000, tol=1e-8, **kw):
        if self.seed is not None: np.random.seed(self.seed)
        lb, ub = self.bounds; D, N = self.dim, self.N
        X = np.random.uniform(lb, ub, (N, D))
        V = np.random.uniform(-0.1, 0.1, (N, D)) * (ub - lb)
        F = np.array([func(x) for x in X])
        pb, pbf = X.copy(), F.copy()
        gi = int(np.argmin(F)); gb, gf = X[gi].copy(), float(F[gi])
        vmax = 0.2 * (ub - lb)
        for t in range(max_iter):
            V = np.clip(0.7 * V + 1.5 * np.random.rand(N, D) * (pb - X)
                        + 1.5 * np.random.rand(N, D) * (gb - X), -vmax, vmax)
            X = np.clip(X + V, lb, ub)
            F = np.array([func(x) for x in X])
            imp = F < pbf; pb[imp], pbf[imp] = X[imp], F[imp]
            gi = int(np.argmin(pbf))
            if pbf[gi] < gf: gb, gf = pb[gi].copy(), float(pbf[gi])
            if abs(gf) < tol: break
        return {"best_solution": gb, "best_fitness": gf}
