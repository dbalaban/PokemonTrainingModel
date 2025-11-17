# bayesian_model.py

from data_structures import *
from PMFs import EV_PMF, IV_PMF
import numpy as np
import math
from typing import Tuple

from tqdm import tqdm

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
            new_W = (m_prior * prior.W + m_upd * upd.W) / denom
            row_sums = new_W.sum(axis=1, keepdims=True)
            new_W = np.divide(new_W, row_sums, out=new_W, where=(row_sums > 0))
        elif mode == "geometric":
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

    k_list = []
    base_term = 2 * B + iv
    for s in s_vals:
        k = s - base_term
        if 0 <= k <= 63:
            k_list.append(k)

    if not k_list:
        return np.zeros(253, dtype=bool)

    return _ev_mask_from_k_set(np.asarray(k_list, dtype=int))

def _predict_stats_batch(EV: np.ndarray, IV: np.ndarray, base: StatBlock, level: int, nature: Nature) -> np.ndarray:
    B = np.array([base.hp, base.atk, base.def_, base.spa, base.spd, base.spe], dtype=int)[:, None]  # (6,1)
    nm = np.array([
        1.0,
        nature.modifier(StatType.ATTACK),
        nature.modifier(StatType.DEFENSE),
        nature.modifier(StatType.SPECIAL_ATTACK),
        nature.modifier(StatType.SPECIAL_DEFENSE),
        nature.modifier(StatType.SPEED),
    ], dtype=float)[:, None]  # (6,1)

    EV4 = EV // 4
    term = ((2 * B + IV + EV4) * level) // 100              # (6, M)

    out = np.empty_like(EV, dtype=int)
    out[0] = term[0] + level + 10
    core = term[1:] + 5
    out[1:] = np.floor(core * nm[1:]).astype(int)
    return out

# --------------------------- Alternating hybrid update ---------------------------

def hybrid_ev_iv_update(
    prior_ev: EV_PMF,
    prior_iv: IV_PMF,
    obs_stats: StatBlock,
    level: int,
    base_stats: StatBlock,
    nature: Nature,
    *,
    mc_particles: int = 20000,
    tol: int = 0,
    max_iters: int = 5,
    iv_tv_tol: float = 1e-4,
) -> Tuple[EV_PMF, IV_PMF]:
    """
    Hybrid alternating update (no corners):
      1) IV update by exact inversion + weighting with current EV marginals.
      2) EV update by importance sampling constrained by current IV posterior.
    """
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

    for _ in tqdm(range(max_iters), desc="Hybrid EV/IV update"):
        # ----- (1) IV update via exact inversion + EV marginals -----
        ev_marg = ev_post.getMarginals(mc_samples=10000)  # (6, 253)
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
            new_iv[s] = (new_iv[s] / row_sum) if row_sum > 0 else old_P[s]

        tv = 0.5 * np.abs(new_iv - old_P).sum(axis=1)
        iv_post.prior = new_iv
        iv_post.normalize_()
        if np.all(tv < iv_tv_tol):
            break

        # ----- (2) EV update via importance sampling (stick-breaking only) -----
        M = mc_particles

        # Sample S5 and W6 from current EV posterior
        S5 = ev_post._sample_S5(M)                         # (5, M)
        W6 = EV_PMF._stick_breaking_from_unit(S5)          # (6, M)

        # Sample totals T
        t_cdf = np.cumsum(ev_post.T)
        uT = ev_post.rng.random(M)
        T_idx = np.searchsorted(t_cdf, uT, side='right').clip(0, ev_post.max_total_ev)

        # Allocate EV by proportional rounding with hard total constraint
        alloc = W6 * T_idx[None, :]
        EV_mat = EV_PMF._round_allocations_to_totals(alloc, T_idx, prior_ev.max_ev)  # (6, M) ints

        # Sample IVs from current IV posterior
        IV_mat = iv_post.sample(M)

        # Likelihood
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

        # Update IV posterior
        iv_post.weighted_add_(IV_mat, w)

        # Rebuild EV posterior (T and W) using weighted particles
        totals = EV_mat.sum(axis=0).clip(0, ev_post.max_total_ev).astype(int)
        new_T = np.bincount(totals, weights=w, minlength=ev_post.max_total_ev + 1)
        new_T = (new_T / new_T.sum()) if new_T.sum() > 0 else ev_post.T.copy()

        # Use sampled S5 to update W (fast + consistent with proposal)
        BINS = ev_post.w_bins
        idx = np.rint(S5 * (BINS - 1)).astype(int).clip(0, BINS - 1)  # (5, M)
        new_W = np.zeros_like(ev_post.W, dtype=float)
        for r in range(5):  # fixed small loop
            new_W[r] = np.bincount(idx[r], weights=w, minlength=BINS)

        row_sums = new_W.sum(axis=1, keepdims=True)
        new_W = np.divide(new_W, row_sums, out=np.zeros_like(new_W), where=(row_sums > 0))

        ev_post = EV_PMF(priorT=new_T, priorW=new_W, w_bins=BINS)

    return ev_post, iv_post

# --------------------------- Single IS update (no alternating) ---------------------------

def update_with_observation(
    prior_ev: EV_PMF,
    prior_iv: IV_PMF,
    obs_stats: StatBlock,
    level: int,
    base_stats: StatBlock,
    nature: Nature,
    M: int = 20000,
    tol: int = 0,
) -> Tuple[EV_PMF, IV_PMF]:
    """
    One-shot importance update with proposal q(EV,IV) = prior_ev × prior_iv.
    Uses stick-breaking sampling only (no corners).
    """
    obs = _sb_to_arr(obs_stats)

    # ----- sample EV from prior_ev via stick-breaking -----
    S5 = prior_ev._sample_S5(M)                         # (5, M)
    W6 = EV_PMF._stick_breaking_from_unit(S5)           # (6, M)

    # sample totals from T
    t_cdf = np.cumsum(prior_ev.T)                       # length 511
    t_u = prior_ev.rng.random(M)
    T_idx = np.searchsorted(t_cdf, t_u, side='right').clip(0, prior_ev.max_total_ev)

    # allocate EV by proportional rounding with hard total
    alloc = W6 * T_idx[None, :]
    EV_mat = EV_PMF._round_allocations_to_totals(alloc, T_idx, prior_ev.max_ev)  # (6, M) ints
    # ----- sample IV from prior_iv -----
    IV_mat = prior_iv.sample(M)                       # (6, M)

    # ----- likelihood weights -----
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

    # ----- IV posterior -----
    new_iv = IV_PMF(prior=prior_iv.P, rng=prior_ev.rng)
    new_iv.weighted_add_(IV_mat, w)

    # ----- EV posterior (T and W) -----
    totals = EV_mat.sum(axis=0).clip(0, prior_ev.max_total_ev).astype(int)
    new_T = np.bincount(totals, weights=w, minlength=prior_ev.max_total_ev + 1)
    new_T = (new_T / new_T.sum()) if new_T.sum() > 0 else prior_ev.T.copy()

    B = prior_ev.w_bins
    new_W = np.zeros_like(prior_ev.W, dtype=float)
    idx = np.rint(S5 * (B - 1)).astype(int).clip(0, B - 1)  # (5, M)
    for r in range(5):
        np.add.at(new_W[r], idx[r], w)
    row_sums = new_W.sum(axis=1, keepdims=True)
    new_W = np.divide(new_W, row_sums, out=np.zeros_like(new_W), where=(row_sums > 0))

    return EV_PMF(priorT=new_T, priorW=new_W, w_bins=B), new_iv