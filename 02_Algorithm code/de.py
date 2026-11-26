import numpy as np

class DE:
    def __init__(self, dim, bounds, pop_size=50, seed=None, F=0.5, CR=0.9):
        self.dim, self.bounds, self.N, self.seed = dim, bounds, pop_size, seed
        self.F, self.CR = F, CR
    def optimize(self, func, max_iter=1000, tol=1e-8, **kw):
        if self.seed is not None: np.random.seed(self.seed)
        lb, ub = self.bounds; D, N = self.dim, self.N
        X = np.random.uniform(lb, ub, (N, D)); F = np.array([func(x) for x in X])
        idx_all = np.arange(N)
        for t in range(max_iter):
            for i in range(N):
                pool = idx_all[idx_all != i]
                r1, r2, r3 = np.random.choice(pool, 3, replace=False)
                v = X[r1] + self.F * (X[r2] - X[r3])
                cr = np.random.rand(D) < self.CR; cr[np.random.randint(D)] = True
                u = np.clip(np.where(cr, v, X[i]), lb, ub)
                fu = func(u)
                if fu < F[i]: X[i], F[i] = u, fu
            if abs(float(F.min())) < tol: break
        i = int(np.argmin(F))
        return {"best_solution": X[i], "best_fitness": float(F[i])}
