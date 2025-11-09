import numpy as np

from random import random
from data_structures import *

from matplotlib import pyplot as plt

class RegimenSimulator:
    def __init__(self, regimen: TrainingRegimen, species: SpeciesInfo, gen: int):
        self.regimen = regimen
        self.species = species
        self.gen = gen
        self.samples = List[StatBlock]()

    def randomEncounter(self, encounters: List[EncounterOption]) -> Encounter:
        rates = np.array([enc.rate for enc in encounters])
        rates = rates / rates.sum()
        opetion_choice = np.random.choice(len(encounters), p=rates)

        level_choice = np.random.randint(
            encounters[opetion_choice].level_min,
            encounters[opetion_choice].level_max + 1
        )

        return Encounter(
            target=encounters[opetion_choice].target,
            level=level_choice
        )

    # Simulate a single training block, returning the resulting experience and EV gains.
    def simulateBlock(self, block: TrainingBlock, exp_start: int) -> tuple[int, StatBlock]:
        # ensure the exp_start is valid for the block's starting level
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
    
    def get_ev_stats(self) -> tuple[StatBlock, StatBlock]:
        ev_array = np.array([[sample.hp, sample.atk, sample.def_, sample.spa, sample.spd, sample.spe] for sample in self.samples])
        means = np.mean(ev_array, axis=0)
        stddevs = np.std(ev_array, axis=0)
        mean_block = StatBlock(*means.astype(int))
        stddev_block = StatBlock(*stddevs.astype(int))
        return mean_block, stddev_block
    
    def plot_ev_distributions(self):
        ev_array = np.array([[sample.hp, sample.atk, sample.def_, sample.spa, sample.spd, sample.spe] for sample in self.samples])
        stat_names = ['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']
        
        # define colors for each stat
        colors = {'HP': 'red', 'Attack': 'blue', 'Defense': 'green',
                  'Special Attack': 'purple', 'Special Defense': 'orange', 'Speed': 'cyan'}

        # exclude stats with zero gain across all samples
        non_zero_indices = np.where(ev_array.sum(axis=0) > 0)[0]
        ev_array = ev_array[:, non_zero_indices]
        stat_names = [stat_names[i] for i in non_zero_indices]
        num_stats = len(stat_names)

        # overlay histograms for each stat
        plt.figure(figsize=(10, 6))
        bins = 30
        for i in range(num_stats):
            plt.hist(ev_array[:, i], bins=bins, alpha=0.5, label=stat_names[i], color=colors[stat_names[i]])
        plt.xlabel('EV Gain')
        plt.ylabel('Frequency')
        plt.title('EV Gain Distributions')
        plt.legend()
        plt.grid(True)
        plt.show()

