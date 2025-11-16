from data_structures import *
from PMFs import EV_PMF
from regimen_sim import RegimenSimulator
import numpy as np

# Regimen table:
# | start lvl | end lvl | location                    | target      | aprx KOs | cum. Speed EVs   | cum. Attack EVs |
# |----------:|--------:|-----------------------------|-------------|---------:|-----------------:|----------------:|
# | 1         | 5       | Route 201                   | Starly      | 7        | 7                | 0               |
# | 5         | 8       | Route 202                   | Shinx       | 6        | 7                | 6               |
# | 8         | 9       | Route 203                   | Starly      | 5        | 12               | 6               |
# | 9         | 10      | Route 203                   | Shinx       | 5        | 12               | 11              |
# | 10        | 12      | Oreburgh Gate / Ravaged Path| Zubat       | 5        | 17               | 11              |
# | 12        | 13      | Route 207                   | Machop      | 5        | 17               | 16              |
# | 13        | 18      | Route 207                   | Ponyta      | 18       | 35               | 16              |
# | 18        | 20      | Route 207                   | Machop      | 18       | 35               | 34              |

Starly = SpeciesInfo(
    name="Starly",
    base_stats=StatBlock(hp=40, atk=55, def_=30, spa=30, spd=30, spe=60),
    ev_yield=StatBlock(hp=0, atk=0, def_=0, spa=0, spd=0, spe=1),
    base_exp_yield=49,
    growth_rate=GrowthRate.MEDIUM_FAST
)

Shinx = SpeciesInfo(
    name="Shinx",
    base_stats=StatBlock(hp=45, atk=65, def_=34, spa=40, spd=34, spe=45),
    ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
    base_exp_yield=53,
    growth_rate=GrowthRate.MEDIUM_FAST
)

Zubat = SpeciesInfo(
    name="Zubat",
    base_stats=StatBlock(hp=40, atk=45, def_=35, spa=30, spd=40, spe=55),
    ev_yield=StatBlock(hp=0, atk=0, def_=0, spa=0, spd=0, spe=1),
    base_exp_yield=49,
    growth_rate=GrowthRate.MEDIUM_FAST
)

Machop = SpeciesInfo(
    name="Machop",
    base_stats=StatBlock(hp=70, atk=80, def_=50, spa=35, spd=35, spe=35),
    ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
    base_exp_yield=61,
    growth_rate=GrowthRate.MEDIUM_FAST
)

Ponyta = SpeciesInfo(
    name="Ponyta",
    base_stats=StatBlock(hp=50, atk=85, def_=55, spa=65, spd=65, spe=90),
    ev_yield=StatBlock(hp=0, atk=0, def_=0, spa=0, spd=0, spe=1),
    base_exp_yield=82,
    growth_rate=GrowthRate.MEDIUM_FAST
)

Riolu = SpeciesInfo(
    name="Riolu",
    base_stats=StatBlock(hp=40, atk=70, def_=40, spa=35, spd=40, spe=60),
    ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
    base_exp_yield=70,
    growth_rate=GrowthRate.MEDIUM_SLOW,
    is_trainer_owned=True
)

regimen = TrainingRegimen(blocks=[
    # Block 1: Level 1-5, Route 201, Starly
    TrainingBlock(
        location="Route 201",
        start_level=1,
        end_level=5,
        encounters=[
            EncounterOption(target=Starly, weight=1.0, levels=[2,3])
        ]
    ),
    # Block 2: Level 5-8, Route 202, Shinx
    TrainingBlock(
        location="Route 202",
        start_level=5,
        end_level=8,
        encounters=[
            EncounterOption(target=Shinx, weight=1.0, levels=[3,4])
        ]
    ),
    # Block 3: Level 8-9, Route 203, Starly
    TrainingBlock(
        location="Route 203",
        start_level=8,
        end_level=9,
        encounters=[
            EncounterOption(target=Starly, weight=0.25, levels=[4,6,7]),
            EncounterOption(target=Starly, weight=0.1, levels=[5])
        ]
    ),
    # Block 4: Level 9-10, Route 203, Shinx
    TrainingBlock(
        location="Route 203",
        start_level=9,
        end_level=10,
        encounters=[
            EncounterOption(target=Shinx, weight=1.0, levels=[4,5])
        ]
    ),
    # Block 5: Level 10-12, Oreburgh Gate / Ravaged Path, Zubat
    TrainingBlock(
        location="Oreburgh Gate",
        start_level=10,
        end_level=12,
        encounters=[
            EncounterOption(target=Zubat, weight=1.0, levels=[5,6,7,8])
        ]
    ),
    # Block 6: Level 12-13, Route 207, Machop
    TrainingBlock(
        location="Route 207",
        start_level=12,
        end_level=13,
        encounters=[
            EncounterOption(target=Machop, weight=1.0, levels=[5,6,7,8])
        ]
    ),
    # Block 7: Level 13-18, Route 207, Ponyta
    TrainingBlock(
        location="Route 207",
        start_level=13,
        end_level=18,
        encounters=[
            EncounterOption(target=Ponyta, weight=1.0, levels=[5,6,7])
        ]
    ),
    # Block 8: Level 18-20, Route 207, Machop
    TrainingBlock(
        location="Route 207",
        start_level=18,
        end_level=20,
        encounters=[
            EncounterOption(target=Machop, weight=1.0, levels=[5,6,7,8])
        ]
    )
])

def main():
    model = RegimenSimulator(
        regimen=regimen,
        species=Riolu,
        gen=4
    )

    num_trials = 10000
    model.run_simulation(num_trials=num_trials, exp_start=0)

    # show histogram of EV distributions
    model.plot_ev_distributions()

    # test the PMF representation
    ev_pmf = model.toPMF()
    marginals = ev_pmf.getMarginals()
    # transform samples from model into marginals
    samples = np.array([[s.hp, s.atk, s.def_, s.spa, s.spd, s.spe] for s in model.samples], dtype=float)
    sample_marginals = []
    for stat_idx in range(6):
        counts, bin_edges = np.histogram(samples[:, stat_idx], bins=range(0, 254), density=True)
        sample_marginals.append(counts)
    sample_marginals = np.array(sample_marginals)  # shape (6, 253)
    # normalize sample marginals to probabilities
    sample_marginals = sample_marginals / sample_marginals.sum(axis=1, keepdims=True)
    # compare
    for stat_idx, stat_name in enumerate(['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']):
        print(f"\nMarginal distribution for {stat_name}:")
        print("EV Value\tPMF Probability\tSample Probability")
        for ev_value in range(253):
            pmf_prob = marginals[stat_idx][ev_value]
            sample_prob = sample_marginals[stat_idx][ev_value]
            if pmf_prob > 0 or sample_prob > 0:
                print(f"{ev_value}\t\t{pmf_prob:.4f}\t\t{sample_prob:.4f}")

if __name__ == "__main__":
    main()