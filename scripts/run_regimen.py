import sys
sys.path.insert(0, '../src')

from data_structures import *
from PMFs import EV_PMF
from regimen_sim import RegimenSimulator
from stat_tracker import track_training_stats
import numpy as np
import argparse

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Pokemon training regimen with stat tracking')
    parser.add_argument('--M', type=int, default=20000,
                        help='Number of Monte Carlo particles for Bayesian updates (default: 20000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--debug-plots', action='store_true',
                        help='Generate matplotlib plots of marginals after each observation')
    parser.add_argument('--smoothing-alpha', type=float, default=0.0,
                        help='EV PMF alpha smoothing parameter (0.0=no smoothing, 0.5=moderate, 1.0=full) (default: 0.0)')
    parser.add_argument('--smoothing-T', type=float, default=0.0,
                        help='EV PMF total smoothing parameter (0.0=no smoothing, 0.5=moderate, 1.0=full) (default: 0.0)')
    parser.add_argument('--update-method', type=str, default='analytic',
                        choices=['analytic', 'hybrid', 'simple'],
                        help='Update method to use (default: analytic). Note: analytic may hang with data inconsistencies; use hybrid with smoothing for robustness.')
    args = parser.parse_args()

    # Define observed stats at levels 12, 19, 20
    # level 12 Riolu stats: 32 24 15 16 15 23
    # level 19 Riolu stats: 46 35 22 23 21 35
    # level 20 Riolu stats: 48 37 23 24 22 36
    observations = [
        ObservedStats(
            level=12,
            stats=StatBlock(hp=32, atk=24, def_=15, spa=16, spd=15, spe=23)
        ),
        ObservedStats(
            level=19,
            stats=StatBlock(hp=46, atk=35, def_=22, spa=23, spd=21, spe=35)
        ),
        ObservedStats(
            level=20,
            stats=StatBlock(hp=48, atk=37, def_=23, spa=24, spd=22, spe=36)
        ),
    ]

    # Define nature (example: neutral nature with no stat modifiers)
    nature = Nature(name="Hardy", inc=None, dec=None)

    print("="*70)
    print("Pokemon Training Regimen Tracker")
    print("="*70)
    print(f"\nSpecies: {Riolu.name}")
    print(f"Nature: {nature.name}")
    print(f"Base Stats: {Riolu.base_stats}")
    print(f"\nTraining Regimen: {len(regimen.blocks)} blocks from level {regimen.blocks[0].start_level} to {regimen.blocks[-1].end_level}")
    print(f"Observations: {len(observations)} measurements at levels {[obs.level for obs in observations]}")
    print(f"\nMonte Carlo particles: {args.M}")
    print(f"Verbose mode: {args.verbose}")
    print(f"Debug plots: {args.debug_plots}")
    print(f"Smoothing alpha: {args.smoothing_alpha}")
    print(f"Smoothing T: {args.smoothing_T}")
    print(f"Update method: {args.update_method}")

    # Run the tracker
    final_ev_pmf, final_iv_pmf = track_training_stats(
        regimen=regimen,
        observations=observations,
        base_stats=Riolu.base_stats,
        nature=nature,
        species_info=Riolu,
        gen=4,
        M=args.M,
        verbose=args.verbose,
        debug_plots=args.debug_plots,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_T=args.smoothing_T,
        update_method=args.update_method,
    )

    print("\n" + "="*70)
    print("Training regimen tracking completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()