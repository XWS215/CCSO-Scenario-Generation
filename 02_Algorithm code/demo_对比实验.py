# -*- coding: utf-8 -*-
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccso import CCSO
from pso import PSO
from sa import SA
from de import DE


def rastrigin(x):
    x = np.asarray(x, float)
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x)))


def schwefel220(x):
    x = np.asarray(x, float)
    return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def griewank(x):
    x = np.asarray(x, float)
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def salomon(x):
    x = np.asarray(x, float)
    r = np.sqrt(np.sum(x ** 2))
    return float(1.0 - np.cos(2 * np.pi * r) + 0.1 * r)


BENCH = {
    'Rastrigin': (rastrigin, -5.12, 5.12,
                  {5: 1.70884, 10: 15.5671}, {5: 0.0, 10: 0.0}),
    'Schwefel220': (schwefel220, -500.0, 500.0,
                    {5: 312.136, 10: 1121.93},
                    {5: -2094.914436362169, 10: -4189.828872724338}),
    'Griewank': (griewank, -600.0, 600.0,
                 {5: 0.0972202, 10: 0.395665}, {5: 0.0, 10: 0.0}),
    'Salomon': (salomon, -100.0, 100.0,
                {5: 0.103896, 10: 0.22837}, {5: 0.0, 10: 0.0}),
}
DIMS = (5, 10)
BUDGET = 5000
RUNS = 5
R_FINE = 0.3 * 0.5
D_MIN = 0.5
METRICS = ['fast', 'hit', 'many', 'eff', 'cov', 'stab']
HIGHER = {'fast': False, 'hit': True, 'many': True, 'eff': True,
          'cov': True, 'stab': True}


class BudgetedFunc:
    def __init__(self, func, lb, ub, f_opt, eps):
        self.func = func
        self.lb, self.ub = lb, ub
        self.thr = f_opt + eps
        self.f_opt = f_opt
        self.n = 0
        self.X = []
        self.F = []
        self.fast = float(BUDGET)

    def __call__(self, x):
        if self.n >= BUDGET:
            return 1e30
        x = np.asarray(x, float).ravel()
        f = float(self.func(x))
        if self.fast == float(BUDGET) and abs(f - self.f_opt) < (self.thr - self.f_opt):
            self.fast = float(self.n + 1)
        self.X.append(x.copy())
        self.F.append(f)
        self.n += 1
        return f


def metrics(bf):
    X = np.asarray(bf.X, float)
    F = np.asarray(bf.F, float)
    fine = []
    for i in range(len(F)):
        if F[i] < bf.thr:
            if not fine:
                fine.append(X[i])
            else:
                d = np.linalg.norm(np.asarray(fine) - X[i], axis=1)
                if d.min() >= R_FINE:
                    fine.append(X[i])
    many = len(fine)
    cov = np.zeros(X.shape[1])
    if fine:
        P = np.asarray(fine)
        for d in range(X.shape[1]):
            nb = max(1, int(round((bf.ub - bf.lb) / D_MIN)))
            idx = np.floor((P[:, d] - bf.lb) / D_MIN).astype(int)
            idx = np.clip(idx, 0, nb - 1)
            cov[d] = len(np.unique(idx)) / float(nb)
    return dict(fast=bf.fast, hit=1.0 if bf.fast < BUDGET else 0.0,
                many=float(many), eff=many / max(1, bf.n), cov=float(cov.mean()))


def make(algo, dim, lb, ub, seed):
    if algo == 'CCSO':
        return CCSO(dim, (lb, ub), pop_size=50, seed=seed, n_solutions=2,
                    n_generals=8, chariot_drive=0.35, sprint_evals=12,
                    sprint_div=0.15, sprint_tol=1e-3, pawn_pull=1.0,
                    task_dist=0.7 * D_MIN)
    if algo == 'PSO':
        return PSO(dim, (lb, ub), pop_size=50, seed=seed)
    if algo == 'SA':
        return SA(dim, (lb, ub), pop_size=50, seed=seed)
    return DE(dim, (lb, ub), pop_size=50, seed=seed)


def run_block(algo, fname, dim, runs):
    func, lb, ub, eps_map, opt_map = BENCH[fname]
    eps, f_opt = eps_map[dim], opt_map[dim]
    rows = []
    for r in range(runs):
        bf = BudgetedFunc(func, lb, ub, f_opt, eps)
        opt = make(algo, dim, lb, ub, r)
        opt.optimize(bf, max_iter=100)
        rows.append(metrics(bf))
    out = {k: float(np.mean([x[k] for x in rows])) for k in ('fast', 'hit', 'many', 'eff', 'cov')}
    m = [x['many'] for x in rows]
    out['stab'] = float(np.clip(1.0 - np.std(m, ddof=1) / np.mean(m), 0, 1)) if np.mean(m) > 0 else 0.0
    return out


def main():
    algos = ['CCSO', 'PSO', 'SA', 'DE']
    blocks = [(f, d) for f in BENCH for d in DIMS]
    table = {}
    rank_rows = []
    metric_ranks = {k: [] for k in METRICS}
    for f, d in blocks:
        per = {}
        for a in algos:
            per[a] = run_block(a, f, d, RUNS)
        table[(f, d)] = per
        for k in METRICS:
            vals = np.array([per[a][k] for a in algos], float)
            rk = stats.rankdata(-vals if HIGHER[k] else vals, method='average')
            rank_rows.append(rk)
            metric_ranks[k].append(rk)
    R = np.asarray(rank_rows)
    avg = R.mean(axis=0)
    print('=' * 74)
    print('CCSO vs PSO/SA/DE   4 benchmarks x D=5,10   %d runs   budget %d' % (RUNS, BUDGET))
    print('=' * 74)
    print('mean distinct critical solutions |H| per block')
    print('%-12s' % 'benchmark' + ''.join('% 10s' % a for a in algos))
    for f, d in blocks:
        print('%-12s' % ('%s D=%d' % (f, d)) + ''.join(
            '% 10.2f' % table[(f, d)][a]['many'] for a in algos))
    print('-' * 74)
    print('%-12s' % 'rank by metric' + ''.join('% 10s' % k for k in METRICS))
    for j, a in enumerate(algos):
        print('%-12s' % a + ''.join(
            '% 10.2f' % np.mean([r[j] for r in metric_ranks[k]]) for k in METRICS))
    print('-' * 74)
    order = np.argsort(avg, kind='stable')
    print('%-12s % 10s' % ('algorithm', 'avg rank'))
    for i in order:
        print('%-12s % 10.2f' % (algos[i], avg[i]))
    print('=' * 74)
    print('first place:', algos[order[0]])
    print('second place:', algos[order[1]])
    print('gap:', round(float(avg[order[1]] - avg[order[0]]), 3))


if __name__ == '__main__':
    main()
