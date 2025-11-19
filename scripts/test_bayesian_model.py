# scripts/test_bayes.py
import sys
sys.path.insert(0, '../src')

import numpy as np
from typing import Tuple
from scipy.stats import entropy, wasserstein_distance

# Project imports
from PMFs import EV_PMF, IV_PMF
from bayesian_model import (
    update_with_observation,
    hybrid_ev_iv_update,
    analytic_update_with_observation,
    _predict_stats_batch,
    feasible_ev_mask_for_stat
)
from data_structures import StatBlock, Nature, StatType

# -----------------------------
# Helpers
# -----------------------------
def rng_ev_vector(rng: np.random.Generator, max_ev=252, max_total=510) -> np.ndarray:
    """
    Random feasible EV vector (length 6, ints in [0,252], sum <= max_total].
    """
    ev = rng.integers(low=0, high=max_ev+1, size=6)
    s = int(ev.sum())
    if s <= max_total:
        return ev
    # downscale proportionally, then fix remainder greedily
    scale = max_total / s
    ev = np.floor(ev * scale).astype(int)
    # spread any leftover budget
    budget = max_total - int(ev.sum())
    if budget > 0:
        order = np.argsort(-ev)  # try to add where it's larger (arbitrary)
        i = 0
        while budget > 0 and i < 6:
            if ev[order[i]] < max_ev:
                ev[order[i]] += 1
                budget -= 1
            i = (i + 1) % 6
    return ev

def sb_to_arr(sb: StatBlock) -> np.ndarray:
    return np.array([sb.hp, sb.atk, sb.def_, sb.spa, sb.spd, sb.spe], dtype=int)

def arr_to_sb(a: np.ndarray) -> StatBlock:
    a = a.astype(int)
    return StatBlock(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]))

def mass_near_ev(marg_row: np.ndarray, ev_star: int, window: int = 1) -> float:
    lo = max(0, ev_star - window)
    hi = min(len(marg_row) - 1, ev_star + window)
    return float(marg_row[lo:hi+1].sum())

def _normalize_safe(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0/len(p))
    return p / s

def _js_divergence(p, q, eps=1e-12):
    # p, q already normalized histograms
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    m = 0.5*(p+q)
    return 0.5*entropy(p, m) + 0.5*entropy(q, m)  # natural log base

def compare_discrete_hist(p_counts, q_counts):
    # Both over same bins. Returns dict with TV, sqrtJS, Hellinger, L1.
    p = _normalize_safe(p_counts)
    q = _normalize_safe(q_counts)
    l1 = float(np.abs(p - q).sum())
    tv = 0.5 * l1
    js = _js_divergence(p, q)
    sqrt_js = float(np.sqrt(js))  # in [0,1]
    # Hellinger
    hell = float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))
    return {"L1": l1, "TV": tv, "sqrtJS": sqrt_js, "Hellinger": hell}

def compare_ev_distributions(ev_true, ev_from_pmf):
    # ev_true, ev_from_pmf: (N,6) integer EV samples
    names = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']
    print("=== Per-stat distribution distances ===")
    for i, name in enumerate(names):
        p_counts, _ = np.histogram(ev_true[:, i], bins=range(0, 254), density=False)
        q_counts, _ = np.histogram(ev_from_pmf[:, i], bins=range(0, 254), density=False)
        d = compare_discrete_hist(p_counts, q_counts)
        # Wasserstein (needs bin positions)
        w1 = wasserstein_distance(np.arange(253), np.arange(253), p_counts, q_counts)
        print(f"{name:10s} TV={d['TV']:.3f}  √JS={d['sqrtJS']:.3f}  Hell={d['Hellinger']:.3f}  W1={w1:.2f} EV")
    # Totals T
    T_true = ev_true.sum(axis=1)
    T_pmf  = ev_from_pmf.sum(axis=1)
    pT, _ = np.histogram(T_true, bins=np.arange(0, 512), density=False)
    qT, _ = np.histogram(T_pmf,  bins=np.arange(0, 512), density=False)
    dT = compare_discrete_hist(pT, qT)
    w1T = wasserstein_distance(np.arange(511), np.arange(511), pT, qT)
    print(f"{'Total EVs':10s} TV={dT['TV']:.3f}  √JS={dT['sqrtJS']:.3f}  Hell={dT['Hellinger']:.3f}  W1={w1T:.2f} EV")

    # Optional: correlation structure check (Pearson)
    C_true = np.corrcoef(ev_true, rowvar=False)
    C_pmf  = np.corrcoef(ev_from_pmf, rowvar=False)
    corr_l1 = np.abs(C_true - C_pmf)[np.triu_indices(6, k=1)].mean()
    print(f"Mean abs diff of off-diag correlations: {corr_l1:.3f}")

def check_feasible_masks(
    base_stats: StatBlock,
    level: int,
    nature: Nature,
    obs_stats: StatBlock,
    *,
    verbose: bool = True,
    max_examples: int = 5,
) -> bool:
    """
    Brute-force test of feasible_ev_mask_for_stat for each stat s and IV in 0..31.
    For each (s, iv), we compute the 'true' feasible set by forward-evaluating
    all EV in 0..252 and matching the observed stat. We compare that set to the
    boolean mask returned by feasible_ev_mask_for_stat.

    Returns True if all masks match exactly; otherwise prints a summary and returns False.
    """
    B = np.array([base_stats.hp, base_stats.atk, base_stats.def_, base_stats.spa, base_stats.spd, base_stats.spe], dtype=int)
    obs = np.array([obs_stats.hp, obs_stats.atk, obs_stats.def_, obs_stats.spa, obs_stats.spd, obs_stats.spe], dtype=int)
    nmult = np.array([
        1.0,
        nature.modifier(StatType.ATTACK),
        nature.modifier(StatType.DEFENSE),
        nature.modifier(StatType.SPECIAL_ATTACK),
        nature.modifier(StatType.SPECIAL_DEFENSE),
        nature.modifier(StatType.SPEED),
    ], dtype=float)

    ev_vals = np.arange(253, dtype=int)            # 0..252
    all_ok = True
    names = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

    for s in range(6):
        is_hp = (s == 0)
        # Vector pieces independent of iv for speed
        ev4 = (ev_vals // 4)                       # (253,)

        for iv in range(32):
            if is_hp:
                # HP: stat = ((2B + iv + floor(EV/4)) * L)//100 + L + 10
                term = ((2 * B[s] + iv + ev4) * level) // 100
                pred = term + level + 10
            else:
                # non-HP: stat = floor( ( ((2B + iv + floor(EV/4)) * L)//100 + 5 ) * n )
                term = ((2 * B[s] + iv + ev4) * level) // 100
                core = term + 5
                pred = np.floor(core * nmult[s]).astype(int)

            brute_mask = (pred == obs[s])          # (253,) bool
            mask = feasible_ev_mask_for_stat(
                y=int(obs[s]),
                B=int(B[s]),
                L=int(level),
                n=float(nmult[s]),
                iv=int(iv),
                is_hp=is_hp,
            )

            # Compare
            if mask.shape != brute_mask.shape:
                if verbose:
                    print(f"[ERROR] Mask shape mismatch for stat {names[s]}, IV={iv}: {mask.shape} vs {brute_mask.shape}")
                all_ok = False
                continue

            diff_fp = np.where(mask & ~brute_mask)[0]    # claimed feasible but not actually feasible
            diff_fn = np.where(~mask & brute_mask)[0]    # missed feasible EVs

            if diff_fp.size or diff_fn.size:
                all_ok = False
                if verbose:
                    print(f"[Mismatch] {names[s]}  IV={iv:2d} | false+={diff_fp.size:3d}, false-={diff_fn.size:3d}")
                    if diff_fp.size:
                        ex = ", ".join(map(str, diff_fp[:max_examples]))
                        print(f"  examples FP (mask True, brute False): {ex}{' ...' if diff_fp.size > max_examples else ''}")
                    if diff_fn.size:
                        ex = ", ".join(map(str, diff_fn[:max_examples]))
                        print(f"  examples FN (mask False, brute True): {ex}{' ...' if diff_fn.size > max_examples else ''}")

    if verbose:
        print("[Feasible-mask check] PASS" if all_ok else "[Feasible-mask check] FAIL")
    return all_ok

# -----------------------------
# Main test
# -----------------------------
def main():
    rng = np.random.default_rng(1337)

    # --- Choose a species "base stats" and level ---
    # If you have a SpeciesInfo handy, you can pull base stats from there.
    # Here we just hardcode a plausible base statline for testing:
    base_stats = StatBlock(80, 95, 85, 65, 65, 90)  # HP, Atk, Def, SpA, SpD, Spe
    level = 50
    nature = Nature(name="neutral", inc=None, dec=None)

    # --- Ground-truth IV* and EV* ---
    iv_star = rng.integers(low=0, high=32, size=6)                  # (0..31)
    ev_star = rng_ev_vector(rng, max_ev=252, max_total=510)          # feasible EVs
    EV_star_mat = ev_star.reshape(6, 1)
    IV_star_mat = iv_star.reshape(6, 1)

    # --- Generate observed stats from the ground truth ---
    obs_stats_arr = _predict_stats_batch(EV_star_mat, IV_star_mat, base_stats, level, nature).reshape(6)
    obs_stats = arr_to_sb(obs_stats_arr)

    # --- Priors ---
    prior_ev = EV_PMF()           # uniform over T and over the 5 stick-breaking rows
    prior_iv = IV_PMF()           # uniform 6x32

    print("=== Ground Truth ===")
    print(f"Level: {level}")
    print(f"Nature multipliers: {[nature.modifier(st) for st in [StatType.ATTACK, StatType.DEFENSE, StatType.SPECIAL_ATTACK, StatType.SPECIAL_DEFENSE, StatType.SPEED]]}")
    print("Base stats:", sb_to_arr(base_stats))
    print("IV*:", iv_star)
    print("EV*:", ev_star)
    print("Observed stats:", obs_stats_arr.tolist())
    print()

    # Sanity Check: Map True EVs to PMF representation and back
    print("=== Sanity Check: EV PMF representation ===")
    ev_star_samples = np.array(ev_star, dtype=float).reshape(1, 6)  # shape (1,6)
    ev_pmf_star = EV_PMF.from_samples(ev_star_samples, allocator="round")
    ev_marginals_star = ev_pmf_star.getMarginals(mc_samples=10000)
    ev_recovered = []
    for stat_idx in range(6):
        counts, bin_edges = np.histogram(ev_star_samples[:, stat_idx], bins=range(0, 254), density=True)
        ev_recovered.append(counts)
    ev_recovered = np.array(ev_recovered)  # shape (6, 253)
    # normalize to probabilities
    ev_recovered = ev_recovered / ev_recovered.sum(axis=1, keepdims=True)
    # compare
    # Report EV marginal mass at ground-truth EVs
    ev_hit_star = [mass_near_ev(ev_marginals_star[s], int(ev_star[s]), window=0) for s in range(6)]
    print("EV* PMF mass at true EV per stat:", [f"{p:.3f}" for p in ev_hit_star])
    print(f"Mean EV mass at truth: {np.mean(ev_hit_star):.3f}")
    print()

    # verify that sampels drawn from ev_pmf_star are always equal to ev_star
    samples_drawn = ev_pmf_star.sample(M=1000)  # shape (1000, 6)
    all_match = np.all(samples_drawn == ev_star.reshape(1, 6))
    print(f"All samples drawn from EV PMF match EV*: {all_match}")
    print()

    print("=== Distribution Recovery Tests ===")
    # generate a random sample of EVs, build and sample PMF, compare to drawn distribution
    num_test_samples = 10000
    ev_test_samples = np.array([rng_ev_vector(rng, max_ev=252, max_total=510) for _ in range(num_test_samples)], dtype=float)  # shape (N,6)
    ev_test_pmf_multinomial = EV_PMF.from_samples(ev_test_samples, allocator="multinomial")
    ev_test_pmf_round = EV_PMF.from_samples(ev_test_samples, allocator="round")
    ev_test_pmf_multinomial_samples = ev_test_pmf_multinomial.sample(M=num_test_samples)  # shape (N,6)
    ev_test_pmf_round_samples = ev_test_pmf_round.sample(M=num_test_samples)      # shape (N,6)
    print("-> multinomial allocator:")
    compare_ev_distributions(ev_test_samples, ev_test_pmf_multinomial_samples)
    print("-> Round allocator:")
    compare_ev_distributions(ev_test_samples, ev_test_pmf_round_samples)
    print()

    print("=== Prior distributions ===")
    # Report prior mass at the ground-truth IVs
    iv_prior_mass = [prior_iv.P[s, iv_star[s]] for s in range(6)]
    print("Prior IV mass at true IV per stat:", [f"{p:.3f}" for p in iv_prior_mass])
    print(f"Mean IV mass at truth: {np.mean(iv_prior_mass):.3f}")
    print()

    # Report prior EV marginal mass near ground-truth
    ev_marg_prior = prior_ev.getMarginals(mc_samples=10000)  # (6, 253)
    ev_prior_hit = [mass_near_ev(ev_marg_prior[s], int(ev_star[s]), window=1) for s in range(6)]
    print("Prior EV mass near true EV per stat:", [f"{p:.3f}" for p in ev_prior_hit])
    print(f"Mean EV mass near truth: {np.mean(ev_prior_hit):.3f}")
    print()

    # --- Feasible-mask correctness check ---
    print("=== Feasible-mask correctness check ===")
    _ = check_feasible_masks(base_stats=base_stats, level=level, nature=nature, obs_stats=obs_stats)
    print()

    # --- Test 1: Single-step importance update ---
    print("=== Test 1: update_with_observation (single IS step) ===")
    post_ev_1, post_iv_1 = update_with_observation(
        prior_ev=prior_ev,
        prior_iv=prior_iv,
        obs_stats=obs_stats,
        level=level,
        base_stats=base_stats,
        nature=nature,
        M=30000,          # more particles -> tighter posterior
        tol=0,            # exact match; consider tol=1 for robustness
    )
    iv_post_1 = post_iv_1.P  # (6, 32)

    # Report IV posterior mass at the ground-truth IVs
    iv_hit = [iv_post_1[s, iv_star[s]] for s in range(6)]
    print("Posterior IV mass at true IV per stat:", [f"{p:.3f}" for p in iv_hit])
    print(f"Mean IV mass at truth: {np.mean(iv_hit):.3f}")

    # Report EV marginal mass near ground-truth
    ev_marg_1 = post_ev_1.getMarginals(mc_samples=10000)  # (6, 253)
    ev_hit = [mass_near_ev(ev_marg_1[s], int(ev_star[s]), window=1) for s in range(6)]
    print("Posterior EV mass near true EV per stat:", [f"{p:.3f}" for p in ev_hit])
    print(f"Mean EV mass near truth: {np.mean(ev_hit):.3f}")
    print()

    # --- Test 2: Hybrid alternating update (IV inversion + EV IS) ---
    print("=== Test 2: hybrid_ev_iv_update (alternating) ===")
    post_ev_2, post_iv_2 = hybrid_ev_iv_update(
        prior_ev=prior_ev,
        prior_iv=prior_iv,
        obs_stats=obs_stats,
        level=level,
        base_stats=base_stats,
        nature=nature,
        mc_particles=20000,
        tol=0,
        max_iters=5,
        iv_tv_tol=1e-4,
    )
    iv_post_2 = post_iv_2.P  # (6, 32)

    iv_hit2 = [iv_post_2[s, iv_star[s]] for s in range(6)]
    print("Posterior IV mass at true IV (hybrid) per stat:", [f"{p:.3f}" for p in iv_hit2])
    print(f"Mean IV mass at truth (hybrid): {np.mean(iv_hit2):.3f}")

    ev_marg_2 = post_ev_2.getMarginals(mc_samples=10000)
    ev_hit2 = [mass_near_ev(ev_marg_2[s], int(ev_star[s]), window=1) for s in range(6)]
    print("Posterior EV mass near true EV per stat:", [f"{p:.3f}" for p in ev_hit2])
    print(f"Mean EV mass near truth (hybrid): {np.mean(ev_hit2):.3f}")

    print("\nDone.")

    # --- Test 3: Analytic Importance Update
    print("=== Test 3: analytic_update_with_observation ===")
    post_ev_3, post_iv_3 = analytic_update_with_observation(
        prior_ev=prior_ev,
        prior_iv=prior_iv,
        obs_stats=obs_stats,
        level=level,
        base_stats=base_stats,
        nature=nature,
        M=100000,          # more particles -> tighter posterior
        verbose=True
    )

    iv_post_3 = post_iv_3.P  # (6, 32)

    # Report IV posterior mass at the ground-truth IVs
    iv_hit3 = [iv_post_3[s, iv_star[s]] for s in range(6)]
    print("Posterior IV mass at true IV per stat:", [f"{p:.3f}" for p in iv_hit3])
    print(f"Mean IV mass at truth: {np.mean(iv_hit3):.3f}")

    # Report EV marginal mass near ground-truth
    ev_marg_3 = post_ev_3.getMarginals(mc_samples=10000)  # (6, 253)
    ev_hit3 = [mass_near_ev(ev_marg_3[s], int(ev_star[s]), window=1) for s in range(6)]
    print("Posterior EV mass near true EV per stat:", [f"{p:.3f}" for p in ev_hit3])
    print(f"Mean EV mass near truth: {np.mean(ev_hit3):.3f}")
    print()

if __name__ == "__main__":
    main()
