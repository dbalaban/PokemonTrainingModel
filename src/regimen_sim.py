#regimen_sim.py

import numpy as np
from typing import List

from data_structures import *  # Encounter, EncounterOption, TrainingRegimen, TrainingBlock, SpeciesInfo, StatBlock
from PMFs import EV_PMF

from matplotlib import pyplot as plt


class RegimenSimulator:
    def __init__(self, regimen: TrainingRegimen, species: SpeciesInfo, gen: int, rng: np.random.Generator | None = None):
        self.regimen = regimen
        self.species = species
        self.gen = gen
        self.rng = rng if rng is not None else np.random.default_rng()
        self.samples: List[StatBlock] = []

    def randomEncounter(self, encounters: List[EncounterOption]) -> Encounter:
        rates = np.array([enc.weight for enc in encounters], dtype=float)
        rates /= rates.sum()
        option_choice = self.rng.choice(len(encounters), p=rates)
        level_choice = self.rng.choice(encounters[option_choice].levels)
        return Encounter(target=encounters[option_choice].target, level=level_choice)

    # Simulate a single training block, returning resulting exp and EV gains.
    def simulateBlock(self, block: TrainingBlock, exp_start: int) -> tuple[int, StatBlock]:
        assert block.start_level == self.species.level_from_exp(exp_start)
        end_exp = self.species.exp_to_level(block.end_level)
        ev_gains = StatBlock(0, 0, 0, 0, 0, 0)
        exp = exp_start
        while exp < end_exp:
            encounter = self.randomEncounter(block.encounters)
            level = self.species.level_from_exp(exp)
            exp_gain = encounter.target.get_exp_yield(encounter.level, level, self.gen)
            ev_gain = encounter.target.ev_yield
            exp += exp_gain
            ev_gains += ev_gain
        # Clamp exp to exact threshold to prevent level overshoot
        exp = min(exp, end_exp)
        return exp, ev_gains

    def simulate_trial(self, exp_start: int = 0) -> StatBlock:
        exp = exp_start
        total_ev_gains = StatBlock(0, 0, 0, 0, 0, 0)
        for block in self.regimen.blocks:
            exp, ev_gains = self.simulateBlock(block, exp)
            total_ev_gains += ev_gains
        return total_ev_gains

    def run_simulation(self, num_trials: int, exp_start: int = 0) -> List[StatBlock]:
        self.samples = []
        for _ in range(num_trials):
            ev_gains = self.simulate_trial(exp_start)
            self.samples.append(ev_gains)
        return self.samples

    def toPMF(self, allocator="multinomial") -> EV_PMF:
        if len(self.samples) == 0:
            raise ValueError("No samples available. Run simulation first.")

        # Pack samples: shape (N, 6)
        ev_array = np.array(
            [[s.hp, s.atk, s.def_, s.spa, s.spd, s.spe] for s in self.samples],
            dtype=float
        )
        
        return EV_PMF.from_samples(ev_array, allocator=allocator)

    def plot_ev_distributions(self):
        ev_array = np.array([[s.hp, s.atk, s.def_, s.spa, s.spd, s.spe] for s in self.samples])
        stat_names = ['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']

        # exclude stats with zero gain across all samples
        non_zero_indices = np.where(ev_array.sum(axis=0) > 0)[0]
        ev_array = ev_array[:, non_zero_indices]
        stat_names = [stat_names[i] for i in non_zero_indices]
        num_stats = len(stat_names)

        plt.figure(figsize=(10, 6))
        bins = 30
        for i in range(num_stats):
            plt.hist(ev_array[:, i], bins=bins, alpha=0.5, label=stat_names[i])
        plt.xlabel('EV Gain')
        plt.ylabel('Frequency')
        plt.title('EV Gain Distributions')
        plt.legend()
        plt.grid(True)
        plt.savefig('ev_distributions.png')
        plt.close()