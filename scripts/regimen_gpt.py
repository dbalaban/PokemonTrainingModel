import math
import random
import numpy as np

# Precompute Medium Slow XP table once
xp_table = [0]*101
for n in range(1, 101):
    if n == 1:
        xp_table[n] = 0
    else:
        xp_table[n] = int((6/5) * n**3 - 15 * n**2 + 100 * n - 140)

def xp_for_level(n: int) -> int:
    return xp_table[n]

# Species config: (ev_type, base_exp, (min_level, max_level))
species_data = {
    'Starly_201': ('spe', 56, (2, 3)),
    'Shinx_202':  ('atk', 60, (3, 4)),
    'Starly_203': ('spe', 56, (4, 6)),
    'Shinx_203':  ('atk', 60, (4, 6)),
    'Zubat_gate': ('spe', 54, (5, 8)),
    'Machop_207': ('atk', 75, (7, 10)),
    'Ponyta_207': ('spe', 152, (7, 10)),
}

regimen = [
    (1, 5,   'Starly_201'),
    (5, 8,   'Shinx_202'),
    (8, 9,   'Starly_203'),
    (9, 10,  'Shinx_203'),
    (10, 12, 'Zubat_gate'),
    (12, 13, 'Machop_207'),
    (13, 18, 'Ponyta_207'),
    (18, 20, 'Machop_207'),
]

N = 500  # modest for reliability without timeout

spe_at_12 = []
atk_at_12 = []
spe_at_19 = []
atk_at_19 = []

for _ in range(N):
    level = 1
    xp = xp_for_level(level)
    spe_ev = 0
    atk_ev = 0
    snap12_done = False
    snap19_done = False

    for (start_lvl, end_lvl, key) in regimen:
        ev_type, base_exp, (min_lv, max_lv) = species_data[key]

        while level < end_lvl:
            foe_lv = random.randint(min_lv, max_lv)
            gain = (base_exp * foe_lv) // 7
            xp += gain

            if ev_type == 'spe':
                spe_ev += 1
            else:
                atk_ev += 1

            # Level-ups
            while level < 100 and xp >= xp_for_level(level + 1):
                level += 1

                if not snap12_done and level == 12:
                    spe_at_12.append(spe_ev)
                    atk_at_12.append(atk_ev)
                    snap12_done = True

                if not snap19_done and level == 19:
                    spe_at_19.append(spe_ev)
                    atk_at_19.append(atk_ev)
                    snap19_done = True

            if level >= end_lvl:
                break

# Convert to numpy arrays for stats
spe12 = np.array(spe_at_12)
atk12 = np.array(atk_at_12)
spe19 = np.array(spe_at_19)
atk19 = np.array(atk_at_19)

summary = {
    "num_trials_with_12": int(len(spe12)),
    "num_trials_with_19": int(len(spe19)),
    "spe12_mean": float(spe12.mean()) if len(spe12) else None,
    "spe12_std": float(spe12.std(ddof=1)) if len(spe12) > 1 else None,
    "atk12_mean": float(atk12.mean()) if len(atk12) else None,
    "atk12_std": float(atk12.std(ddof=1)) if len(atk12) > 1 else None,
    "spe19_mean": float(spe19.mean()) if len(spe19) else None,
    "spe19_std": float(spe19.std(ddof=1)) if len(spe19) > 1 else None,
    "atk19_mean": float(atk19.mean()) if len(atk19) else None,
    "atk19_std": float(atk19.std(ddof=1)) if len(atk19) > 1 else None,
}
summary

