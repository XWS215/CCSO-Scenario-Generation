import numpy as np

def sample_distinct(N, k):
    draws = np.empty((N, k), dtype=int)
    ex = np.arange(N).reshape(-1, 1)
    for j in range(k):
        idx = np.random.randint(0, N - ex.shape[1], N)
        r = idx.copy()
        for c in range(ex.shape[1]):
            r += (r >= ex[:, c])
        draws[:, j] = r
        ex = np.sort(np.concatenate([ex, r.reshape(-1, 1)], axis=1), axis=1)
    return draws


def bound_in_box(v, lb, ub):
    v = np.asarray(v, dtype=float)
    lo = v < lb
    if lo.any(): v = np.where(lo, lb + (lb - v), v)
    hi = v > ub
    if hi.any(): v = np.where(hi, ub - (v - ub), v)
    out = (v < lb) | (v > ub)
    if out.any(): v = np.where(out, np.random.uniform(lb, ub), v)
    return v


def basin_refine(func, x, f0, lb, ub, delta, max_evals, floor_rel=2e-3):
    D = x.size; best, fb = x.copy(), float(f0)
    step = 0.05 * delta * np.ones(D); floor = floor_rel * delta; ev = 0
    while ev + 2 * D <= max_evals and step.max() > floor:
        any_move = False
        for i in range(D):
            if step[i] <= floor: continue
            improved = False
            for sgn in (1.0, -1.0):
                while ev < max_evals:
                    y = best.copy()
                    y[i] = float(min(max(best[i] + sgn * step[i], lb), ub))
                    if y[i] == best[i]: break
                    fy = func(y); ev += 1
                    if fy < fb: best, fb, improved, any_move = y, fy, True, True
                    else: break
            if not improved: step[i] *= 0.5
        if not any_move: break
    h = floor
    while ev + 3 * D <= max_evals:
        moved = False
        for i in range(D):
            if ev + 3 > max_evals: break
            xa = best.copy(); xa[i] = float(min(max(best[i] - h, lb), ub))
            xc = best.copy(); xc[i] = float(min(max(best[i] + h, lb), ub))
            fa = func(xa); fc = func(xc); ev += 2
            den = fa - 2.0 * fb + fc
            if den <= 0.0: continue
            d = float(np.clip(h * (fa - fc) / (2.0 * den), -2.0 * h, 2.0 * h))
            if d == 0.0: continue
            y = best.copy(); y[i] = float(min(max(best[i] + d, lb), ub))
            if y[i] == best[i]: continue
            fy = func(y); ev += 1
            if fy < fb: best, fb, moved = y, fy, True
        if not moved:
            h *= 0.25
            if h < 1e-4 * floor: break
    return best, fb, ev


def archive_update(arch_X, arch_F, X, F, niche_r, cap):
    pX = np.vstack([arch_X, X]) if len(arch_X) else X
    pF = np.concatenate([arch_F, F]) if len(arch_F) else F
    kX, kF = [], []
    for i in np.argsort(pF):
        if len(kX) >= cap: break
        if all(float(np.linalg.norm(pX[i] - z)) >= niche_r for z in kX):
            kX.append(pX[i]); kF.append(float(pF[i]))
    if not kX:
        return pX[:0], pF[:0]
    return np.array(kX), np.asarray(kF)


def select_diverse(pool_X, pool_F, k, radius, min_dist):
    order = np.argsort(pool_F)
    P, Pf = pool_X[order], pool_F[order]
    n = len(P)
    if n == 0: return P[:0], Pf[:0]
    c1, c2 = max(1, int(0.5 * n)), max(1, int(0.8 * n))
    bands = [list(range(0, c1)), list(range(c1, c2)), list(range(c2, n))]
    chosen = [0]; r = float(radius)
    while len(chosen) < k:
        pick, pick_d = -1, -1.0
        for band in bands:
            for i in band:
                if i in chosen: continue
                d = min(float(np.linalg.norm(P[i] - P[j])) for j in chosen)
                if d >= max(r, min_dist) and d > pick_d:
                    pick, pick_d = i, d
            if pick >= 0: break
        if pick < 0:
            r *= 0.6
            if r < min_dist: break
            continue
        chosen.append(pick)
    return P[chosen], Pf[chosen]


def assemble_solutions(func, pool_X, pool_F, k, niche_r, min_dist, lb, ub, delta):
    D = pool_X.shape[1]; ev = 0
    seeds_X, seeds_F = select_diverse(pool_X, pool_F, k, niche_r, min_dist)
    acc_X, acc_F = [], []
    for x, f in zip(seeds_X, seeds_F):
        if all(float(np.linalg.norm(x - z)) >= min_dist for z in acc_X):
            acc_X.append(x); acc_F.append(float(f))
    while len(acc_X) < k and len(pool_X):
        A = np.array(acc_X)
        d = np.min(np.linalg.norm(pool_X[:, None, :] - A[None, :, :], axis=-1), axis=1)
        ok = np.where(d >= min_dist)[0]
        if len(ok) == 0: break
        far = ok[np.argsort(-d[ok])[:max(1, len(ok) // 5)]]
        take = far[int(np.argmin(pool_F[far]))]
        acc_X.append(pool_X[take]); acc_F.append(float(pool_F[take]))
    guard = 0
    while len(acc_X) < k and acc_X and guard < 60 * k:
        guard += 1
        base = acc_X[np.random.randint(len(acc_X))]
        v = np.random.randn(D); v /= (np.linalg.norm(v) + 1e-12)
        for scale in (2.4, 1.6, 1.05):
            y = bound_in_box(base + v * min_dist * scale, lb, ub)
            if all(float(np.linalg.norm(y - z)) >= min_dist for z in acc_X):
                acc_X.append(y); acc_F.append(float(func(y))); ev += 1; break
    return np.array(acc_X), np.array(acc_F), ev


class CCSOAblation:
    def __init__(self, dim, bounds, pop_size=50, n_generals=6, seed=None,
                 n_solutions=8, min_dist_ratio=0.03, min_dist_abs=0.35,
                 niche_ratio=0.05, multi_dim_max=5.0, task_dist=None,
                 chariot_drive=0.25, sprint_evals=12, sprint_div=0.15,
                 sprint_tol=1e-3, pawn_pull=1.0, ablate=None):
        self.dim = dim; self.lb, self.ub = bounds
        self.delta = self.ub - self.lb
        self.N = pop_size; self.K = n_generals; self.seed = seed
        self.WIN_BASIN = 8; self.WIN_BETTER = 25; self.STAG_CHECK = 40
        self.RESTART_EPS = 1e-3; self.RESTART_DROUGHT = 4
        self.RESTART_FRAC = 0.5; self.RESTART_SCALE = 0.008
        self.n_solutions = int(min(50, max(1, n_solutions)))
        self.min_dist_ratio = float(min_dist_ratio)
        self.min_dist_abs = float(min_dist_abs)
        self.multi_dim_max = float(multi_dim_max)
        self.niche_ratio = float(niche_ratio)
        self.task_dist = None if task_dist is None else float(task_dist)
        self.chariot_drive = float(np.clip(chariot_drive, 0.0, 1.0))
        self.sprint_on = True
        self.sprint_evals = int(sprint_evals)
        self.sprint_div = float(sprint_div)
        self.sprint_tol = float(sprint_tol)
        self.pawn_pull = float(pawn_pull)
        ab = set(ablate or ())
        if "roles" in ab: ab |= {"chariot", "horse", "cannon", "pawn"}
        if "opening" in ab: self.chariot_drive = 0.0
        if "roles" in ab or "sprint" in ab or "chariot_only" in ab:
            self.sprint_on = False
        if "all" in ab:
            ab |= {"chariot", "horse", "cannon", "pawn", "endgame", "opening"}
            self.chariot_drive = 0.0; self.sprint_on = False
        if "only_opening" in ab: ab.add("phases_op")
        if "only_midgame" in ab: ab.add("phases")
        if "only_endgame" in ab: ab.add("phases_en")
        _ROLE = {"only_chariot": 0, "only_horse": 1, "only_cannon": 2, "only_pawn": 3}
        self._only_role = next((v for k, v in _ROLE.items() if k in ab), None)
        self._only_general = bool("only_general" in ab)
        if self._only_role is not None:
            self.sprint_on = False
        if self._only_general:
            ab |= {"chariot", "horse", "cannon", "pawn"}
        self.dbg = None
        self.ablate = ab
        self.off_roles = {r for r, nm in enumerate(("chariot", "horse", "cannon", "pawn"))
                          if nm in ab}

    def optimize(self, func, max_iter=1000, tol=1e-8, n_solutions=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        K_sol = (self.n_solutions if n_solutions is None
                 else int(min(50, max(1, n_solutions))))
        multi = K_sol > 1
        N, D, K = self.N, self.dim, self.K
        T = max_iter
        lb, ub, delta = self.lb, self.ub, self.delta
        idx_all = np.arange(N)
        if self._only_general:
            is_piece = np.zeros(N, dtype=bool)
        elif self._only_role is not None:
            is_piece = np.ones(N, dtype=bool)
        else:
            is_piece = idx_all >= K

        X = np.random.uniform(lb, ub, (N, D))
        F = np.array([func(x) for x in X])
        if self._only_role is not None:
            roles = np.full(N, self._only_role, dtype=int)
        else:
            roles = (np.zeros(N, dtype=int) if "chariot_only" in self.ablate
                     else np.random.choice(4, N, p=[0.2, 0.2, 0.4, 0.2]))
        cannon_mode = np.random.randint(0, 2, N)
        chariot_style = np.random.randint(0, 2, N)
        stagnation = np.zeros(N, dtype=np.int32)
        hunt_axis = np.full(N, -1); hunt_win = np.zeros(N); hunt_dir = np.zeros((N, D))

        best_idx = int(np.argmin(F))
        gbest = X[best_idx].copy(); gbest_f = float(F[best_idx])
        div_max = 0.0; progress = 0.0
        pending_skip = 0; gbest_drought = 0; ev_debt = 0.0
        sprint_last = np.full(D, np.inf)
        arch_X = np.zeros((0, D)); arch_F = np.zeros(0)
        arch_cap = max(2 * K_sol, 8) if multi else 0
        ref_r = (self.task_dist if self.task_dist is not None
                 else self.niche_ratio * delta)
        niche_r = ref_r * (2.0 / max(2, D))
        arch_every = 5
        restart_drought = 0
        n_arch_prev = 0; gbest_ph = float(gbest_f)
        q_new = 0; q_any = 0

        for t in range(T):
            if pending_skip > 0: pending_skip -= 1; continue
            g0 = gbest_f
            order = np.argsort(F)
            X, F = X[order], F[order]
            roles = roles[order]; stagnation = stagnation[order]
            hunt_axis = hunt_axis[order]; hunt_win = hunt_win[order]
            hunt_dir = hunt_dir[order]
            cannon_mode = cannon_mode[order]; chariot_style = chariot_style[order]
            if F[0] < gbest_f:
                gbest_f = float(F[0]); gbest = X[0].copy()

            half = max(1, N // 2)
            div = float(np.mean(np.std(X[:half], axis=0)) / (delta + 1e-12))
            if t < self.WIN_BASIN: div_max = max(div_max, div)
            progress = max(progress, 1.0 - div / max(div_max, 1e-12), t / max(1, T))

            new_basin = len(arch_X) > n_arch_prev
            better = gbest_f < gbest_ph - 1e-9 * max(1.0, abs(gbest_ph))
            n_arch_prev = len(arch_X); gbest_ph = gbest_f
            q_new = 0 if new_basin else q_new + 1
            q_any = 0 if (new_basin or better) else q_any + 1
            ph_state = (0 if q_new < self.WIN_BASIN
                        else (1 if q_any < self.WIN_BETTER else 2))
            ph_budget = 0 if t < T / 3 else (1 if t < 2 * T / 3 else 2)
            ph = max(ph_state, ph_budget)
            if "phases" in self.ablate: ph = 1
            if "phases_op" in self.ablate: ph = 0
            if "phases_en" in self.ablate: ph = 2
            ph_op = ph
            if "opening" in self.ablate: ph_op = max(ph_op, 1)
            if "midgame" in self.ablate: ph_op = 0 if ph_op == 1 else ph_op
            if "endgame" in self.ablate: ph_op = min(ph_op, 1)

            mutants = np.zeros_like(X)
            R5 = sample_distinct(N, 5)
            e_scale = 0.002 * delta * (1.0 - progress) ** 2
            anchors = (np.vstack([gbest[None, :], arch_X]) if len(arch_X)
                       else gbest[None, :])
            anchor = anchors[np.arange(N) % len(anchors)]
            role_masks = [np.zeros(N, dtype=bool) for _ in range(4)]
            full_move = np.zeros(N, bool)
            chariot_scan_dir = np.zeros((N, D))
            chariot_was_p1 = np.zeros(N, dtype=bool)

            if self.off_roles:
                F_fb = (0.5 + 0.4 * np.random.rand(N))[:, None]
                mutants = (anchor + F_fb * (X[R5[:, 1]] - X[R5[:, 2]])
                           + e_scale * np.random.uniform(-1.0, 1.0, (N, D)))

            gmask = (np.ones(N, dtype=bool) if self._only_general
                     else np.zeros(N, dtype=bool) if self._only_role is not None
                     else idx_all < K)
            if gmask.any():
                n_g = int(gmask.sum())
                if "general" in self.ablate:
                    F_g = (0.5 + 0.4 * np.random.rand(n_g))[:, None]
                    mutants[gmask] = (anchor[gmask]
                                      + F_g * (X[R5[gmask, 1]] - X[R5[gmask, 2]])
                                      + e_scale * np.random.uniform(-1.0, 1.0, (n_g, D)))
                elif ph_op == 0:
                    mutants[gmask] = gbest + 0.05 * delta * np.random.uniform(
                        -1.0, 1.0, (n_g, D))
                elif ph_op == 1:
                    ag = anchor[gmask]
                    F1 = (0.5 * np.random.rand(n_g))[:, None]
                    F2 = (0.5 * np.random.rand(n_g))[:, None]
                    mutants[gmask] = (ag + F1 * (X[R5[gmask, 1]] - X[R5[gmask, 2]])
                                      + F2 * (X[R5[gmask, 3]] - X[R5[gmask, 4]])
                                      + e_scale * np.random.uniform(-1.0, 1.0, (n_g, D)))
                else:
                    mutants[gmask] = (anchor[gmask]
                                      + niche_r * np.random.standard_normal((n_g, D)))

            cmask = is_piece & (roles == 0)
            if cmask.any() and "chariot" not in self.ablate:
                c_idx = np.where(cmask)[0]; n_c = len(c_idx)
                v = np.empty((n_c, D)); dirs = np.zeros((n_c, D))
                if ph_op == 0:
                    v = anchor[c_idx].copy()
                    drive = np.random.rand(n_c) < self.chariot_drive
                    if drive.any():
                        v[drive] = np.random.uniform(lb, ub, (int(drive.sum()), D))
                        full_move[c_idx[drive]] = True
                    ax_i = np.where(~drive)[0]
                    if len(ax_i):
                        d1_ax = np.random.randint(0, D, len(ax_i))
                        v[ax_i, d1_ax] = np.random.uniform(lb, ub, len(ax_i))
                        dirs[ax_i, d1_ax] = 1.0
                    p1 = np.ones(n_c, dtype=bool)
                elif ph_op == 2:
                    v = anchor[c_idx] + niche_r * np.random.standard_normal((n_c, D))
                    p1 = np.zeros(n_c, dtype=bool)
                else:
                    c_hunt_active = (hunt_axis[c_idx] >= 0) & (hunt_win[c_idx] > 0)
                    p1 = ~c_hunt_active
                    if p1.any():
                        p1_pos = np.where(p1)[0]
                        v[p1] = anchor[c_idx][p1]
                        use_line = chariot_style[c_idx][p1] == 1
                        ax_pos = p1_pos[~use_line]
                        if len(ax_pos) > 0:
                            d1_ax = np.random.randint(0, D, len(ax_pos))
                            v[ax_pos, d1_ax] = np.random.uniform(lb, ub, len(ax_pos))
                            dirs[ax_pos, d1_ax] = 1.0
                        ln_pos = p1_pos[use_line]
                        if len(ln_pos) > 0:
                            d_hat = np.random.standard_normal((len(ln_pos), D))
                            d_hat /= (np.linalg.norm(d_hat, axis=1, keepdims=True) + 1e-300)
                            t_ln = np.random.uniform(-0.5 * delta, 0.5 * delta, len(ln_pos))
                            v[ln_pos] = v[ln_pos] + t_ln[:, None] * d_hat
                            dirs[ln_pos] = d_hat
                    p2 = c_hunt_active
                    if p2.any():
                        p2_idx = c_idx[p2]
                        win = hunt_win[p2_idx][:, None]
                        v[p2] = (X[p2_idx] + np.random.uniform(-1.0, 1.0, (int(p2.sum()), 1))
                                 * win * hunt_dir[p2_idx])
                mutants[c_idx] = v
                chariot_scan_dir[c_idx] = dirs
                chariot_was_p1[c_idx] = p1
                role_masks[0] = cmask

            kmask = is_piece & (roles == 1)
            if kmask.any() and "horse" not in self.ablate:
                n_k = int(kmask.sum())
                v = anchor[kmask].copy()
                if D >= 2:
                    d1 = np.random.randint(0, D, n_k)
                    d2 = (d1 + 1 + np.random.randint(0, D - 1, n_k)) % D
                    s1 = np.random.choice([-1, 1], n_k); s2 = np.random.choice([-1, 1], n_k)
                    step = 0.1 * delta * (1.0 - 0.35 * ph_op)
                    v[np.arange(n_k), d1] += s1 * 2 * step
                    v[np.arange(n_k), d2] += s2 * step
                    if ph_op == 2 and len(arch_X):
                        leg = anchor[kmask].copy()
                        leg[np.arange(n_k), d1] += s1 * step
                        dmin = np.min(np.linalg.norm(
                            leg[:, None, :] - arch_X[None, :, :], axis=2), axis=1)
                        blocked = np.where(dmin < niche_r)[0]
                        if len(blocked) > 0:
                            dj = np.random.randint(0, D, len(blocked))
                            v[blocked] = (X[np.where(kmask)[0][blocked]]
                                          + np.random.uniform(-1.0, 1.0, (len(blocked), D))
                                          * (0.5 * step))
                            v[blocked, dj] += np.random.choice(
                                [-1.0, 1.0], len(blocked)) * 2.0 * step
                mutants[kmask] = v
                role_masks[1] = kmask

            pmask = is_piece & (roles == 3)
            if pmask.any() and "pawn" not in self.ablate:
                n_p = int(pmask.sum())
                F_p = (0.5 + 0.3 * np.random.rand(n_p))[:, None] * self.pawn_pull
                F_q = (0.5 + 0.5 * np.random.rand(n_p))[:, None]
                if ph_op == 0:
                    mutants[pmask] = X[pmask] + F_p * (anchor[pmask] - X[pmask])
                elif ph_op == 1:
                    v = (X[pmask] + F_p * (anchor[pmask] - X[pmask])
                         + F_q * (X[R5[pmask, 1]] - X[R5[pmask, 2]]))
                    lat = np.random.rand(n_p) < 0.3
                    if lat.any():
                        li = np.where(lat)[0]
                        dj = np.random.randint(0, D, len(li))
                        v[li, dj] = anchor[pmask][li, dj]
                    mutants[pmask] = v
                else:
                    v = X[pmask].copy()
                    dj = np.random.randint(0, D, n_p)
                    v[np.arange(n_p), dj] += (
                        (anchor[pmask][np.arange(n_p), dj] - v[np.arange(n_p), dj])
                        * (0.5 + 0.5 * np.random.rand(n_p)))
                    mutants[pmask] = v
                role_masks[3] = pmask

            nmask = is_piece & (roles == 2)
            if nmask.any() and "cannon" not in self.ablate:
                n_n = int(nmask.sum())
                v = np.empty((n_n, D))
                if ph_op == 0:
                    C1 = np.random.uniform(-1.0, 1.0, (n_n, D))
                    v = anchor[nmask] + C1 * (anchor[nmask] - X[nmask])
                elif ph_op == 1:
                    if gbest_drought == 0:
                        C0 = np.random.uniform(0.0, 0.35, (n_n, D))
                        v = X[nmask] + C0 * (gbest - X[nmask])
                    else:
                        m0 = cannon_mode[nmask] == 0
                        if m0.any():
                            C0 = np.random.uniform(0.0, 1.0, (int(m0.sum()), D))
                            v[m0] = C0 * anchor[nmask][m0]
                        m1 = ~m0
                        if m1.any():
                            C1 = np.random.uniform(-1.0, 1.0, (int(m1.sum()), D))
                            v[m1] = (anchor[nmask][m1]
                                     + C1 * (anchor[nmask][m1] - X[nmask][m1]))
                else:
                    v = anchor[nmask] + niche_r * np.random.standard_normal((n_n, D))
                scale = 0.1 * delta * (1.0 - 0.6 * progress)
                tail = np.random.rand(n_n) < 0.2
                if tail.any():
                    d_t = np.random.randint(0, D, int(tail.sum()))
                    v[tail, d_t] += (np.random.standard_cauchy(int(tail.sum())) * scale)
                mutants[nmask] = v
                role_masks[2] = nmask

            mutants = np.clip(mutants, lb, ub)

            if D // 2 <= 1:
                m = np.zeros((N, D), bool)
                m[np.arange(N), np.random.randint(0, D, N)] = True
                trials = np.where(m, mutants, X)
            else:
                _idx = np.argsort(np.random.rand(N, D), axis=1)[:, :D // 2]
                m = np.zeros((N, D), bool)
                m[np.arange(N)[:, None], _idx] = True
                trials = np.where(m, mutants, X)
            trials[full_move] = mutants[full_move]

            if self.dbg is not None and 'g' not in self.dbg:
                self.dbg['g'] = int(gmask.sum())
                self.dbg['r'] = [int(m.sum()) for m in role_masks]
                self.dbg['n'] = int(N)

            F_trials = np.array([func(x) for x in trials])
            improve = F_trials < F
            meaningful = F_trials < F - 1e-4 * np.maximum(1e-6, np.abs(F))
            X[improve] = trials[improve]
            F[improve] = F_trials[improve]
            stagnation[meaningful] = 0
            stagnation[~meaningful] += 1

            if multi and (t % arch_every == 0 or t == T - 1):
                arch_X, arch_F = archive_update(arch_X, arch_F, X, F,
                                                 niche_r, arch_cap)

            cmask_u = role_masks[0]
            if cmask_u.any():
                c_idx = np.where(cmask_u)[0]
                was_p1 = chariot_was_p1[c_idx]; succ = meaningful[c_idx]
                open_hunt = was_p1 & succ
                hunt_axis[c_idx[open_hunt]] = 1
                hunt_dir[c_idx[open_hunt]] = chariot_scan_dir[c_idx][open_hunt]
                hunt_win[c_idx[open_hunt]] = 0.05 * delta
                tighten = (~was_p1) & succ
                hunt_win[c_idx[tighten]] *= 0.5
                widen = (~was_p1) & ~succ
                hunt_win[c_idx[widen]] *= 1.5
                abandon = widen & (hunt_win[c_idx] > 0.5 * delta)
                hunt_axis[c_idx[abandon]] = -1; hunt_win[c_idx[abandon]] = 0.0
                hunt_dir[c_idx[abandon]] = 0.0

            check = ((stagnation == self.STAG_CHECK) & is_piece
                     if "stagnation" not in self.ablate
                     else np.zeros(N, dtype=bool))
            if check.any():
                ch = np.where(check)[0]
                if "chariot_only" not in self.ablate and self._only_role is None:
                    roles[ch] = np.random.randint(0, 4, len(ch))
                cannon_mode[ch] = np.random.randint(0, 2, len(ch))
                chariot_style[ch] = np.random.randint(0, 2, len(ch))
                n_ch = len(ch)
                cand = np.random.uniform(lb, ub, (n_ch, 8, D))
                if len(arch_X):
                    dd = np.min(np.linalg.norm(
                        cand[:, :, None, :] - arch_X[None, None, :, :], axis=3), axis=2)
                    pick = np.argmax(dd, axis=1)
                else:
                    pick = np.random.randint(0, 8, n_ch)
                X[ch] = cand[np.arange(n_ch), pick]
                F[ch] = np.array([func(x) for x in X[ch]])
                stagnation[ch] = 0
                hunt_axis[ch] = -1; hunt_win[ch] = 0.0; hunt_dir[ch] = 0.0

            if gbest_f < g0 - self.RESTART_EPS * max(1.0, abs(g0)):
                restart_drought = 0
            else:
                restart_drought += 1
            if (progress > 0.85 and restart_drought >= self.RESTART_DROUGHT
                    and "endgame" not in self.ablate):
                n_rs = max(2, int(round(N * self.RESTART_FRAC)))
                order_rs = np.argsort(F); idx_rs = order_rs[-n_rs:]
                X[idx_rs] = np.clip(gbest + np.random.uniform(-1, 1, (n_rs, D))
                                    * (self.RESTART_SCALE * delta), lb, ub)
                F[idx_rs] = np.array([func(x) for x in X[idx_rs]])
                stagnation[idx_rs] = 0
                restart_drought = 0

            if (self.sprint_on and t > 0 and div < self.sprint_div * div_max
                    and (gbest != sprint_last).any()):
                sprint_last = gbest.copy()
                xs, fs_, ev_s = basin_refine(
                    func, gbest, gbest_f, lb, ub, delta,
                    max(int(self.sprint_evals), 6 * D),
                    floor_rel=self.sprint_tol)
                if fs_ < gbest_f:
                    gbest, gbest_f = xs, fs_
                    X[0], F[0] = xs, fs_
                    gbest_drought = 0
                ev_debt += ev_s
                if ev_debt >= N:
                    n_sk = int(ev_debt // N); ev_debt -= n_sk * N
                    pending_skip = max(pending_skip, n_sk)

            gbest_drought = 0 if gbest_f < g0 else gbest_drought + 1
            if abs(gbest_f) < tol: break

        min_dist = max(self.min_dist_ratio * delta, self.min_dist_abs)
        pool_X = np.vstack([X, arch_X]) if len(arch_X) else X
        pool_F = np.concatenate([F, arch_F]) if len(arch_F) else F
        sol_X, sol_F, sol_ev = assemble_solutions(
            func, pool_X, pool_F, K_sol, niche_r, min_dist, lb, ub, delta)

        return {"best_solution": gbest.copy(), "best_fitness": float(gbest_f),
                "solutions": sol_X, "solution_fitness": sol_F,
                "n_solutions": int(len(sol_X)), "assembly_evals": int(sol_ev),
                "final_progress": float(progress),
                "archive_X": arch_X.copy(), "archive_F": arch_F.copy()}

ABLATIONS = {
    "full": None, "general_only": ["only_general"], "chariot_only": ["only_chariot"],
    "horse_only": ["only_horse"], "cannon_only": ["only_cannon"], "pawn_only": ["only_pawn"],
    "opening_only": ["only_opening"], "midgame_only": ["only_midgame"], "endgame_only": ["only_endgame"],
}


def build_variant(name, dim, bounds, **kwargs):
    kwargs = dict(kwargs)
    kwargs["ablate"] = ABLATIONS[name]
    return CCSOAblation(dim, bounds, **kwargs)
