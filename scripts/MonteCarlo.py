import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# Mechanics
# -----------------------------

# Medium Slow XP formula (Gen 4)
def xp_for_level(n: int) -> int:
    if n == 1:
        return 0
    # Official Medium Slow formula:
    # EXP(n) = (6/5)n^3 - 15n^2 + 100n - 140
    return int((6/5) * n**3 - 15 * n**2 + 100 * n - 140)

# Species config: (ev_type, base_exp, min_level, max_level)
# ev_type: 'spe' for Speed EV, 'atk' for Attack EV
species_data = {
    "Starly_201": ("spe", 56, 2, 3),
    "Shinx_202":  ("atk", 60, 3, 4),
    "Starly_203": ("spe", 56, 4, 6),
    "Shinx_203":  ("atk", 60, 4, 6),
    "Zubat_gate": ("spe", 54, 5, 8),
    "Machop_207": ("atk", 75, 7, 10),
    "Ponyta_207": ("spe", 152, 7, 10),
}

# Training regimen:
# You stay in each row until you HIT the end level.
# You KO ONLY the listed target; run from everything else.
regimen = [
    (1, 5,   "Starly_201"),
    (5, 8,   "Shinx_202"),
    (8, 9,   "Starly_203"),
    (9, 10,  "Shinx_203"),
    (10, 12, "Zubat_gate"),
    (12, 13, "Machop_207"),
    (13, 18, "Ponyta_207"),
    (18, 20, "Machop_207"),  # we'll snapshot at 19 during this row
]

# -----------------------------
# Single simulation
# -----------------------------

def simulate_one():
    level = 1
    xp = xp_for_level(level)
    spe_ev = 0
    atk_ev = 0

    spe_at_12 = None
    atk_at_12 = None
    spe_at_19 = None
    atk_at_19 = None

    for start_lvl, end_lvl, key in regimen:
        ev_type, base_exp, min_lv, max_lv = species_data[key]

        # If we overshot this block's end level in a previous block, skip
        if level >= end_lvl:
            continue

        while level < end_lvl:
            foe_lv = random.randint(min_lv, max_lv)
            gain = (base_exp * foe_lv) // 7
            if gain <= 0:
                raise RuntimeError("Non-positive EXP gain (check data)")

            xp += gain

            # EV gain for KO
            if ev_type == "spe":
                spe_ev += 1
            elif ev_type == "atk":
                atk_ev += 1

            # Handle potential multiple level-ups from one KO
            # (Ponyta in particular can jump you more than one level)
            while level < 100 and xp >= xp_for_level(level + 1):
                level += 1

                # Snapshot EXACTLY when we HIT level 12
                if level == 12 and spe_at_12 is None:
                    spe_at_12 = spe_ev
                    atk_at_12 = atk_ev

                # Snapshot EXACTLY when we HIT level 19
                if level == 19 and spe_at_19 is None:
                    spe_at_19 = spe_ev
                    atk_at_19 = atk_ev

            if level >= end_lvl:
                break

    return spe_at_12, atk_at_12, spe_at_19, atk_at_19

# -----------------------------
# Monte Carlo
# -----------------------------

def run_monte_carlo(num_trials=20000):
    spe12_vals = []
    atk12_vals = []
    spe19_vals = []
    atk19_vals = []

    for _ in range(num_trials):
        spe12, atk12, spe19, atk19 = simulate_one()

        # It's possible (though extremely unlikely with this regimen)
        # to miss hitting exactly 12 or 19 due to weird jumps.
        # Only keep trials where we DID hit those levels.
        if spe12 is not None and atk12 is not None:
            spe12_vals.append(spe12)
            atk12_vals.append(atk12)
        if spe19 is not None and atk19 is not None:
            spe19_vals.append(spe19)
            atk19_vals.append(atk19)

    return (
        np.array(spe12_vals),
        np.array(atk12_vals),
        np.array(spe19_vals),
        np.array(atk19_vals),
    )

# -----------------------------
# Helper: print stats & plot hist
# -----------------------------

def summarize_and_plot(vals, title, stat_name):
    if vals.size == 0:
        print(f"{title}: no data")
        return

    mean = vals.mean()
    std = vals.std(ddof=1)
    print(f"{title} ({stat_name} EVs):")
    print(f"  n = {len(vals)}")
    print(f"  mean = {mean:.2f}")
    print(f"  std  = {std:.2f}")
    print(f"  min  = {vals.min()}, max = {vals.max()}")
    counts = Counter(vals)
    print("  histogram (EV : count):")
    for ev in sorted(counts):
        print(f"    {ev:3d} : {counts[ev]}")
    print()

    plt.figure()
    plt.hist(vals, bins=range(vals.min(), vals.max() + 2), align="left", rwidth=0.8)
    plt.xlabel(f"{stat_name} EVs")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(range(vals.min(), vals.max() + 1))
    plt.show()

if __name__ == "__main__":
    from collections import Counter

    spe12, atk12, spe19, atk19 = run_monte_carlo(num_trials=20000)

    summarize_and_plot(spe12, "Level 12 Speed EVs", "Speed")
    summarize_and_plot(atk12, "Level 12 Attack EVs", "Attack")
    summarize_and_plot(spe19, "Level 19 Speed EVs", "Speed")
    summarize_and_plot(atk19, "Level 19 Attack EVs", "Attack")
