# bayesian_model.py

from data_structures import *
from PMFs import EV_PMF, IV_PMF, barycentric_simplex
import numpy as np
import math
from typing import Tuple

def nature_to_multipliers(nature: Nature) -> np.ndarray:
    return np.array([
        1.0,  # HP is unaffected
        nature.modifier(StatType.ATTACK),
        nature.modifier(StatType.DEFENSE),
        nature.modifier(StatType.SPECIAL_ATTACK),
        nature.modifier(StatType.SPECIAL_DEFENSE),
        nature.modifier(StatType.SPEED),
    ], dtype=float)

def update_ev_pmf(prior: EV_PMF, upd: EV_PMF, mode: str = "linear") -> EV_PMF:
    """
    Update EV PMF:
      - T: discrete convolution (totals add)
      - W: barycenter of the five independent stick-breaking row pmfs,
           weighted solely by E[T] from prior and update.
    mode: "linear" or "geometric"
    """
    # ---- 1) Update T via convolution ----
    # both vectors length = max_total_ev+1
    new_T = np.convolve(prior.T, upd.T)[:prior.max_total_ev + 1]
    s = new_T.sum()
    new_T = new_T / s if s > 0 else prior.T.copy()

    # ---- 2) Update W using only expected totals as masses ----
    tvals_prior = np.arange(prior.max_total_ev + 1, dtype=float)
    tvals_upd   = np.arange(upd.max_total_ev   + 1, dtype=float)
    m_prior = float((tvals_prior * prior.T).sum())
    m_upd   = float((tvals_upd   * upd.T  ).sum())
    denom = m_prior + m_upd

    if denom <= 0:
        new_W = prior.W.copy()
    else:
        if mode == "linear":
            # Dirichlet-mean / convex barycenter per row
            new_W = (m_prior * prior.W + m_upd * upd.W) / denom
            row_sums = new_W.sum(axis=1, keepdims=True)
            new_W = np.divide(new_W, row_sums, out=new_W, where=(row_sums > 0))
        elif mode == "geometric":
            # Product-of-experts barycenter per row (sharper)
            eps = 1e-12
            logW = m_prior * np.log(prior.W + eps) + m_upd * np.log(upd.W + eps)
            new_W = np.exp(logW)
            row_sums = new_W.sum(axis=1, keepdims=True)
            new_W = np.divide(new_W, row_sums, out=new_W, where=(row_sums > 0))
        else:
            raise ValueError("mode must be 'linear' or 'geometric'")

    return EV_PMF(priorT=new_T, priorW=new_W, w_bins=prior.w_bins)

def _sb_to_arr(sb: StatBlock) -> np.ndarray:
    return np.array([sb.hp, sb.atk, sb.def_, sb.spa, sb.spd, sb.spe], dtype=int)

def _t_candidates(y: int, n: float) -> range:
    """
    Integers t such that floor(t * n) == y (non-HP stats).
    """
    lo = math.ceil(y / n)
    # subtract a tiny epsilon at the top to avoid including the next integer
    hi = math.floor((y + 1 - 1e-9) / n)
    if hi < lo:
        return range(0, 0)
    return range(lo, hi + 1)

def _feasible_s_interval(u: int, L: int) -> tuple[int, int]:
    """
    s = 2B + IV + k must satisfy s in [ceil(100*u/L), ceil(100*(u+1)/L)-1]
    """
    s_lo = math.ceil(100 * u / L)
    s_hi = math.ceil(100 * (u + 1) / L) - 1
    return s_lo, s_hi

def _ev_mask_from_k_set(k_set: np.ndarray) -> np.ndarray:
    """
    Given feasible k (0..63), produce a boolean mask over EV=0..252
    where EV in union_k [4k,4k+3].
    """
    mask = np.zeros(253, dtype=bool)
    for k in k_set:
        a = max(0, 4 * k)
        b = min(252, 4 * k + 3)
        if a <= b:
            mask[a:b + 1] = True
    return mask

def feasible_ev_mask_for_stat(
    y: int, B: int, L: int, n: float, iv: int, is_hp: bool
) -> np.ndarray:
    """
    Return a boolean mask over EV in [0..252] feasible for a given IV, observed y,
    base B, level L, and nature multiplier n (1.0 for HP).
    """
    # Gather feasible s values (for non-HP via t-candidates; for HP directly)
    s_vals = set()

    if is_hp:
        u = y - L - 10
        s_lo, s_hi = _feasible_s_interval(u, L)
        for s in range(s_lo, s_hi + 1):
            s_vals.add(s)
    else:
        for t in _t_candidates(y, n):
            u = t - 5
            s_lo, s_hi = _feasible_s_interval(u, L)
            for s in range(s_lo, s_hi + 1):
                s_vals.add(s)

    if not s_vals:
        return np.zeros(253, dtype=bool)

    # For each s, k = s - (2B + IV), must be in [0..63]
    k_list = []
    base_term = 2 * B + iv
    for s in s_vals:
        k = s - base_term
        if 0 <= k <= 63:
            k_list.append(k)

    if not k_list:
        return np.zeros(253, dtype=bool)

    return _ev_mask_from_k_set(np.asarray(k_list, dtype=int))

def _predict_stats_batch(EV: np.ndarray, IV: np.ndarray,
                         base: StatBlock, level: int, nature: Nature) -> np.ndarray:
    B = np.array([base.hp, base.atk, base.def_, base.spa, base.spd, base.spe], dtype=int)
    nm = np.array([
        1.0,
        nature.modifier(StatType.ATTACK),
        nature.modifier(StatType.DEFENSE),
        nature.modifier(StatType.SPECIAL_ATTACK),
        nature.modifier(StatType.SPECIAL_DEFENSE),
        nature.modifier(StatType.SPEED),
    ], dtype=float)

    out = np.empty_like(EV, dtype=int)
    hp_term = ((2 * B[0] + IV[0] + (EV[0] // 4)) * level) // 100
    out[0] = hp_term + level + 10
    for s in range(1, 6):
        term = ((2 * B[s] + IV[s] + (EV[s] // 4)) * level) // 100
        core = term + 5
        out[s] = np.floor(core * nm[s]).astype(int)
    return out


def _round_allocations_to_total(ev_alloc: np.ndarray, desired_total: int, max_ev: int) -> np.ndarray:
    """
    Given a float allocation vector `ev_alloc` (length 6) whose exact sum equals
    `desired_total` up to floating error, produce an integer vector in [0,max_ev]
    whose sum equals `desired_total` by floor+largest-remainder rounding while
    respecting per-stat bounds.
    """
    # floor and remainders
    ev_floor = np.floor(ev_alloc).astype(int)
    ev_floor = np.clip(ev_floor, 0, max_ev)
    rem = int(desired_total - ev_floor.sum())
    if rem == 0:
        return ev_floor

    remainders = ev_alloc - np.floor(ev_alloc)
    # sort indices by descending fractional part
    idx_order = np.argsort(-remainders)
    out = ev_floor.copy()
    # try to add 1 to highest remainders while respecting max_ev
    for i in idx_order:
        if rem <= 0:
            break
        if out[i] < max_ev:
            out[i] += 1
            rem -= 1

    # if still remaining (rare, e.g., all at max_ev), distribute to any indices (wrap)
    if rem > 0:
        for i in range(6):
            if rem <= 0:
                break
            if out[i] < max_ev:
                add = min(max_ev - out[i], rem)
                out[i] += add
                rem -= add

    # if rem still > 0, we couldn't satisfy desired_total due to bounds; as a last
    # resort, adjust by clipping and leave sum as close as possible
    return out

def hybrid_ev_iv_update(
    prior_ev: EV_PMF,
    prior_iv: IV_PMF,          # shape (6, 32) rows sum to 1
    obs_stats: StatBlock,
    level: int,
    base_stats: StatBlock,
    nature: Nature,
    *,
    mc_particles: int = 20000,     # IS particles for EV correction step
    tol: int = 0,                  # per-stat tolerance in likelihood (0 exact, 1 for ±1)
    max_iters: int = 5,
    iv_tv_tol: float = 1e-4,       # stop if IV TV distance per row is small
) -> Tuple[EV_PMF, IV_PMF]:
    """
    Hybrid alternating update:
      1) IV update by exact inversion + weighting with current EV marginals.
      2) EV update by importance sampling constrained by current IV posterior.
    Repeats until IV posterior stabilizes or max_iters is reached.

    Returns: (posterior_EV_PMF, posterior_IV_6x32)
    """
    # Initialize
    ev_post = prior_ev
    iv_post = IV_PMF(prior=prior_iv.P, rng=prior_ev.rng)

    obs = np.array([obs_stats.hp, obs_stats.atk, obs_stats.def_, obs_stats.spa, obs_stats.spd, obs_stats.spe], dtype=int)
    B = np.array([base_stats.hp, base_stats.atk, base_stats.def_, base_stats.spa, base_stats.spd, base_stats.spe], dtype=int)
    nmult = np.array([
        1.0,
        nature.modifier(StatType.ATTACK),
        nature.modifier(StatType.DEFENSE),
        nature.modifier(StatType.SPECIAL_ATTACK),
        nature.modifier(StatType.SPECIAL_DEFENSE),
        nature.modifier(StatType.SPEED),
    ], dtype=float)

    for it in range(max_iters):
        # ----- (1) IV update via exact inversion + EV marginals -----
        ev_marg = ev_post.getMarginals(mc_samples=10000)  # (6,253)
        old_P = iv_post.P.copy()
        new_iv = np.zeros_like(old_P)

        for s in range(6):
            is_hp = (s == 0)
            for iv_val in range(32):
                mask = feasible_ev_mask_for_stat(
                    y=int(obs[s]), B=int(B[s]), L=int(level),
                    n=float(nmult[s]), iv=iv_val, is_hp=is_hp,
                )
                mass = ev_marg[s, mask].sum()
                new_iv[s, iv_val] = old_P[s, iv_val] * mass

            row_sum = new_iv[s].sum()
            if row_sum > 0:
                new_iv[s] /= row_sum
            else:
                new_iv[s] = old_P[s]

        # convergence check on IV (TV distance)
        tv = 0.5 * np.abs(new_iv - old_P).sum(axis=1)
        iv_post.prior = new_iv          # or iv_post.P = new_iv
        iv_post.normalize_()
        if np.all(tv < iv_tv_tol):
            break

        # ----- (2) EV update via importance sampling -----
        M = mc_particles

        S5 = ev_post._sample_S5(M)
        W6 = EV_PMF._stick_breaking_from_unit(S5)
        t_cdf = np.cumsum(ev_post.T)
        uT = ev_post.rng.random(M)
        T_idx = np.searchsorted(t_cdf, uT, side='right').clip(0, ev_post.max_total_ev)

        uniqT, inv = np.unique(T_idx, return_inverse=True)
        C_cache = {int(t): EV_PMF._get_corners(int(t)).astype(float) for t in uniqT}

        EV_mat = np.empty((6, M), dtype=float)
        for k, t in enumerate(uniqT):
            sel = (inv == k)
            # _get_corners returns rows = vertices; use transpose so columns are
            # vertices when multiplying by weight vectors from W6. The float
            # allocations are stored in EV_mat and will be rounded per-sample
            # to enforce exact totals.
            EV_mat[:, sel] = C_cache[int(t)].T @ W6[:, sel]

        # Round per-sample allocations to integer EVs while enforcing per-sample
        # totals equal to the sampled T_idx values (and per-stat bounds).
        EV_int = np.empty((6, M), dtype=int)
        for j in range(M):
            desired_t = int(T_idx[j])
            EV_int[:, j] = _round_allocations_to_total(EV_mat[:, j], desired_t, ev_post.max_ev)
        EV_mat = EV_int

        IV_mat = iv_post.sample(M)  # sample from current IV posterior

        pred = _predict_stats_batch(EV_mat, IV_mat, base_stats, level, nature)
        if tol == 0:
            w = (pred == obs[:, None]).all(axis=0).astype(float)
        else:
            w = (np.abs(pred - obs[:, None]) <= tol).all(axis=0).astype(float)

        Z = w.sum()
        if Z == 0:
            err = np.abs(pred - obs[:, None]).sum(axis=0)
            lam = 1.0
            w = np.exp(-lam * err); Z = w.sum()
            w = (w / Z) if Z > 0 else np.full(M, 1.0 / M)
        else:
            w /= Z

        # Update IV posterior from weighted samples
        iv_post.weighted_add_(IV_mat, w)

        # Rebuild EV posterior (T and W)
        totals = EV_mat.sum(axis=0)
        # rounding of allocations can sometimes produce a total of max_total_ev+1
        # (e.g., 510 -> 511) due to per-stat rounding; clip to valid index range
        totals_clipped = np.clip(totals, 0, ev_post.max_total_ev).astype(int)
        new_T = np.zeros_like(ev_post.T, dtype=float)
        np.add.at(new_T, totals_clipped, w)
        new_T = new_T / new_T.sum() if new_T.sum() > 0 else ev_post.T.copy()

        new_W = np.zeros_like(ev_post.W, dtype=float)
        BINS = ev_post.w_bins
        # Precompute the corners columns used for each sample (based on sampled T)
        Ccols_per_sample = [C_cache[int(uniqT[inv[j]])].T for j in range(M)]

        for j in range(M):
            C_cols = Ccols_per_sample[j]
            w6 = barycentric_simplex(C_cols, EV_mat[:, j].astype(float))
            s5 = EV_PMF._invert_stick_breaking(w6)
            idx = np.rint(s5 * (BINS - 1)).astype(int).clip(0, BINS - 1)
            new_W[np.arange(5), idx] += w[j]

        # normalize rows
        row_sums = new_W.sum(axis=1, keepdims=True)
        new_W = np.divide(new_W, row_sums, out=np.zeros_like(new_W), where=(row_sums > 0))

        ev_post = EV_PMF(priorT=new_T, priorW=new_W, w_bins=BINS)

    return ev_post, iv_post

def update_with_observation(
    prior_ev: EV_PMF,
    prior_iv: IV_PMF,                 # shape (6, 32), each row sums to 1
    obs_stats: StatBlock,                 # observed stats at 'level'
    level: int,
    base_stats: StatBlock,                # species base stats
    nature: Nature,                       # shape (6,), values in {0.9,1.0,1.1}
    M: int = 20000,                       # particles
    tol: int = 0,                         # per-stat tolerance (0 exact; 1 allows ±1)
) -> Tuple[EV_PMF, IV_PMF]:
    """
    Importance update with proposal q(EV,IV) = prior_ev × prior_iv.
    Normalized weights are proportional to P(obs | EV, IV).
    Returns (posterior_EV_PMF, posterior_IV_6x32).
    """
    nature_multipliers = nature_to_multipliers(nature)

    obs = _sb_to_arr(obs_stats)
    base = _sb_to_arr(base_stats).astype(int)
    nm = nature_multipliers.astype(float)

    # ----- sample EV from prior_ev -----
    S5 = prior_ev._sample_S5(M)                          # (5, M)
    W6 = EV_PMF._stick_breaking_from_unit(S5)            # (6, M)

    # sample totals from T
    t_cdf = np.cumsum(prior_ev.T)                        # length 511
    t_u = prior_ev.rng.random(M)
    t_idx = np.searchsorted(t_cdf, t_u, side='right').clip(0, prior_ev.max_total_ev)

    # corners cache
    uniq_T, inv = np.unique(t_idx, return_inverse=True)
    C_cache = {int(t): EV_PMF._get_corners(int(t)).astype(float) for t in uniq_T}

    EV_mat = np.empty((6, M), dtype=float)
    for k, t in enumerate(uniq_T):
        sel = (inv == k)
        # use transpose of corners so columns correspond to vertex allocations
        EV_mat[:, sel] = C_cache[int(t)].T @ W6[:, sel]

    EV_int = np.empty((6, M), dtype=int)
    for j in range(M):
        desired_t = int(t_idx[j])
        EV_int[:, j] = _round_allocations_to_total(EV_mat[:, j], desired_t, prior_ev.max_ev)
    EV_mat = EV_int

    IV_mat = prior_iv.sample(M)                       # (6, M)
    pred = _predict_stats_batch(EV_mat, IV_mat, base_stats, level, nature)

    # ----- likelihood weights -----
    if tol == 0:
        w = (pred == obs[:, None]).all(axis=0).astype(float)
    else:
        w = (np.abs(pred - obs[:, None]) <= tol).all(axis=0).astype(float)

    Z = w.sum()
    if Z == 0:
        # soft fallback (prevents degeneracy)
        err = np.abs(pred - obs[:, None]).sum(axis=0)
        lam = 1.0
        w = np.exp(-lam * err)
        Z = w.sum()
        w = (w / Z) if Z > 0 else np.full(M, 1.0 / M)
    else:
        w /= Z

    # ----- IV posterior (6 x 32) -----
    new_iv = IV_PMF(prior=prior_iv.P, rng=prior_ev.rng)
    new_iv.weighted_add_(IV_mat, w)

    # ----- EV posterior as EV_PMF (T and W rows) -----
    totals = EV_mat.sum(axis=0)
    # clip totals to valid index range (rounding can push some totals to max+1)
    totals_clipped = np.clip(totals, 0, prior_ev.max_total_ev).astype(int)
    new_T = np.zeros_like(prior_ev.T, dtype=float)
    np.add.at(new_T, totals_clipped, w)
    new_T = new_T / new_T.sum() if new_T.sum() > 0 else prior_ev.T.copy()

    new_W = np.zeros_like(prior_ev.W, dtype=float)
    B = prior_ev.w_bins
    # Precompute corner-columns per sample (based on sampled totals uniq_T/inv)
    Ccols_per_sample = [C_cache[int(uniq_T[inv[j]])].T.astype(float) for j in range(M)]

    for j in range(M):
        C_cols = Ccols_per_sample[j]
        w6 = barycentric_simplex(C_cols, EV_mat[:, j].astype(float))
        s5 = EV_PMF._invert_stick_breaking(w6)
        idx = np.rint(s5 * (B - 1)).astype(int).clip(0, B - 1)
        new_W[np.arange(5), idx] += w[j]

    # now normalize rows
    row_sums = new_W.sum(axis=1, keepdims=True)
    new_W = np.divide(new_W, row_sums, out=np.zeros_like(new_W), where=(row_sums > 0))

    # IV posterior via IV_PMF
    new_iv = IV_PMF(prior=prior_iv.P, rng=prior_ev.rng)
    new_iv.weighted_add_(IV_mat, w)

    return EV_PMF(priorT=new_T, priorW=new_W, w_bins=B), new_iv