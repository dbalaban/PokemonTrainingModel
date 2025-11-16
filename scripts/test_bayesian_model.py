# scripts/test_bayes.py
import numpy as np
from typing import Tuple

# Project imports
from PMFs import EV_PMF, IV_PMF
from bayesian_model import (
    update_with_observation,
    hybrid_ev_iv_update,
    _predict_stats_batch,  # re-use the model's stat calculation
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
    nature = Nature(name="nuetral", inc=None, dec=None)

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
        tol=1,            # exact match; consider tol=1 for robustness
    )
    iv_post_1 = post_iv_1.P  # (6, 32)

    # Report IV posterior mass at the ground-truth IVs
    iv_hit = [iv_post_1[s, iv_star[s]] for s in range(6)]
    print("Posterior IV mass at true IV per stat:", [f"{p:.3f}" for p in iv_hit])
    print(f"Mean IV mass at truth: {np.mean(iv_hit):.3f}")

    # Report EV marginal mass near ground-truth (±1)
    ev_marg_1 = post_ev_1.getMarginals(mc_samples=10000)  # (6, 253)
    ev_hit = [mass_near_ev(ev_marg_1[s], int(ev_star[s]), window=1) for s in range(6)]
    print("Posterior EV mass near true EV (±1) per stat:", [f"{p:.3f}" for p in ev_hit])
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
    print("Posterior EV mass near true EV (±1, hybrid) per stat:", [f"{p:.3f}" for p in ev_hit2])
    print(f"Mean EV mass near truth (hybrid): {np.mean(ev_hit2):.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
