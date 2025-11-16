import numpy as np
from typing import List

from data_structures import *  # Encounter, EncounterOption, TrainingRegimen, TrainingBlock, SpeciesInfo, StatBlock
from PMFs import EV_PMF
from PMFs import barycentric_simplex

from matplotlib import pyplot as plt

class RegimenSimulator:
    def __init__(self, regimen: TrainingRegimen, species: SpeciesInfo, gen: int, w_bins: int = 506):
        self.regimen = regimen
        self.species = species
        self.gen = gen
        self.samples: List[StatBlock] = []
        self.w_bins = w_bins

    def randomEncounter(self, encounters: List[EncounterOption]) -> Encounter:
        rates = np.array([enc.weight for enc in encounters], dtype=float)
        rates /= rates.sum()
        option_choice = np.random.choice(len(encounters), p=rates)
        level_choice = np.random.choice(encounters[option_choice].levels)
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

    def toPMF(self) -> EV_PMF:
        if len(self.samples) == 0:
            raise ValueError("No samples available. Run simulation first.")

        # Pack samples: shape (N, 6)
        ev_array = np.array([[s.hp, s.atk, s.def_, s.spa, s.spd, s.spe] for s in self.samples], dtype=float)

        max_ev = 252
        n_stats = 6
        max_total_ev = 2 * max_ev + n_stats  # 510
        B = self.w_bins
        eps = 1e-12

        # ---- 1) PMF over totals T ----
        total_evs = ev_array.sum(axis=1).astype(int)
        T_hist, _ = np.histogram(total_evs, bins=np.arange(0, max_total_ev + 2), density=True)
        T_hist = T_hist / T_hist.sum()  # safety

        # ---- 2) PMFs over stick-breaking variables (5 rows, B bins) ----
        W_counts = np.zeros((5, B), dtype=float)

        for sample in ev_array:
            T = int(round(sample.sum()))
            # get_corners returns (6,6) rows=vertices; we want columns=vertices for the solver
            C_cols = EV_PMF._get_corners(T).T  # (6,6)

            # Find barycentric weights w (>=0, sum=1) so that C_cols @ w ≈ sample
            w = barycentric_simplex(C_cols, sample)  # shape (6,)

            # Invert stick-breaking to get s1..s5 in [0,1]
            S = EV_PMF._invert_stick_breaking(w, eps=eps)  # (5,)

            # Bin each s_k to nearest bin center i/(B-1)
            idx = np.rint(S * (B - 1)).astype(int)
            idx = np.clip(idx, 0, B - 1)

            W_counts[np.arange(5), idx] += 1.0

        # Row-wise normalize to get 5 independent pmfs over [0,1]
        row_sums = W_counts.sum(axis=1, keepdims=True)
        W_hist = np.divide(W_counts, row_sums, out=np.zeros_like(W_counts), where=(row_sums > 0))

        return EV_PMF(priorT=T_hist, priorW=W_hist)

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
