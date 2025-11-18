# stat_tracker.py

"""
End-to-end IV/EV tracking over training regimen and observations.

This module provides functions to track Pokemon stats through a training regimen
by combining:
  - TrainingRegimen and RegimenSimulator for simulating EV gains
  - IV_PMF and EV_PMF for maintaining probability distributions
  - analytic_update_with_observation for Bayesian updates from observed stats
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from data_structures import (
    TrainingRegimen,
    TrainingBlock,
    ObservedStats,
    StatBlock,
    Nature,
)
from PMFs import EV_PMF, IV_PMF
from regimen_sim import RegimenSimulator
from bayesian_model import analytic_update_with_observation


def split_regimen_at_levels(
    regimen: TrainingRegimen,
    levels: List[int],
) -> TrainingRegimen:
    """
    Split regimen blocks so that each level in `levels` occurs at a block boundary.
    
    If a level falls strictly between a block's start_level and end_level, that block
    is split into multiple blocks at that level.
    
    Parameters
    ----------
    regimen : TrainingRegimen
        The original training regimen to split
    levels : List[int]
        List of levels where splits should occur
        
    Returns
    -------
    TrainingRegimen
        A new regimen with blocks split at the specified levels
    """
    if not levels:
        return regimen
    
    # Sort levels for consistent processing
    split_levels = sorted(set(levels))
    
    new_blocks = []
    for block in regimen.blocks:
        # Find all split points within this block (strictly between start and end)
        splits_in_block = [
            lvl for lvl in split_levels
            if block.start_level < lvl < block.end_level
        ]
        
        if not splits_in_block:
            # No splits needed for this block
            new_blocks.append(block)
            continue
        
        # Sort splits and add start/end levels to create intervals
        all_levels = sorted([block.start_level] + splits_in_block + [block.end_level])
        
        # Create a new block for each interval
        for i in range(len(all_levels) - 1):
            start = all_levels[i]
            end = all_levels[i + 1]
            
            # Only create block if there's a non-zero interval
            if start < end:
                new_block = TrainingBlock(
                    start_level=start,
                    end_level=end,
                    location=block.location,
                    encounters=block.encounters,
                    trainer_owned=block.trainer_owned,
                )
                new_blocks.append(new_block)
    
    return TrainingRegimen(blocks=new_blocks)


def print_iv_histograms(iv_pmf: IV_PMF, title: str = "IV Marginal Distributions") -> None:
    """
    Print histograms of IV marginal distributions for each stat.
    
    Parameters
    ----------
    iv_pmf : IV_PMF
        The IV PMF to print histograms for
    title : str
        Title to print above the histograms
    """
    stat_names = ['HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed']
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    for s_idx, stat_name in enumerate(stat_names):
        print(f"\n{stat_name}:")
        marginal = iv_pmf.P[s_idx]  # Shape (32,)
        
        # Find the range with non-zero probability
        nonzero_indices = np.where(marginal > 1e-6)[0]
        if len(nonzero_indices) == 0:
            print("  (All probabilities near zero)")
            continue
        
        min_iv = int(nonzero_indices[0])
        max_iv = int(nonzero_indices[-1])
        
        # Print histogram
        for iv in range(min_iv, max_iv + 1):
            prob = marginal[iv]
            if prob > 1e-6:
                bar_length = int(prob * 50)  # Scale to 50 chars max
                bar = '█' * bar_length
                print(f"  IV {iv:2d}: {prob:6.4f} {bar}")


def print_ev_histograms(ev_pmf: EV_PMF, title: str = "EV Marginal Distributions") -> None:
    """
    Print histograms of EV marginal distributions for each stat.
    
    Parameters
    ----------
    ev_pmf : EV_PMF
        The EV PMF to print histograms for
    title : str
        Title to print above the histograms
    """
    stat_names = ['HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed']
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    # Get marginals via sampling
    marginals = ev_pmf.getMarginals(mc_samples=10000)  # Shape (6, 253)
    
    for s_idx, stat_name in enumerate(stat_names):
        print(f"\n{stat_name}:")
        marginal = marginals[s_idx]
        
        # Find the range with non-zero probability
        nonzero_indices = np.where(marginal > 1e-6)[0]
        if len(nonzero_indices) == 0:
            print("  (All probabilities near zero)")
            continue
        
        min_ev = int(nonzero_indices[0])
        max_ev = int(nonzero_indices[-1])
        
        # For large ranges, bin the EVs
        if max_ev - min_ev > 30:
            # Print binned histogram (groups of 8)
            bin_size = 8
            for start_ev in range(min_ev, max_ev + 1, bin_size):
                end_ev = min(start_ev + bin_size - 1, max_ev)
                bin_prob = marginal[start_ev:end_ev + 1].sum()
                if bin_prob > 1e-6:
                    bar_length = int(bin_prob * 50)
                    bar = '█' * bar_length
                    print(f"  EV {start_ev:3d}-{end_ev:3d}: {bin_prob:6.4f} {bar}")
        else:
            # Print individual EVs
            for ev in range(min_ev, max_ev + 1):
                prob = marginal[ev]
                if prob > 1e-6:
                    bar_length = int(prob * 50)
                    bar = '█' * bar_length
                    print(f"  EV {ev:3d}: {prob:6.4f} {bar}")


def plot_marginals(
    ev_pmf: EV_PMF,
    iv_pmf: IV_PMF,
    title: str = "IV/EV Marginals",
) -> None:
    """
    Plot IV and EV marginal distributions using matplotlib.
    
    This function is optional and will fail gracefully if matplotlib is not available
    or if running in a headless environment.
    
    Parameters
    ----------
    ev_pmf : EV_PMF
        The EV PMF to plot
    iv_pmf : IV_PMF
        The IV PMF to plot
    title : str
        Title for the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        stat_names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        
        # Create figure with 2 rows (IV and EV)
        fig, axes = plt.subplots(2, 6, figsize=(18, 8))
        fig.suptitle(title, fontsize=16)
        
        # Plot IV marginals
        for s_idx in range(6):
            ax = axes[0, s_idx]
            marginal = iv_pmf.P[s_idx]
            iv_values = np.arange(32)
            ax.bar(iv_values, marginal, width=1.0, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('IV Value')
            ax.set_ylabel('Probability')
            ax.set_title(f'{stat_names[s_idx]} IV')
            ax.set_xlim(-0.5, 31.5)
        
        # Plot EV marginals
        ev_marginals = ev_pmf.getMarginals(mc_samples=5000)
        for s_idx in range(6):
            ax = axes[1, s_idx]
            marginal = ev_marginals[s_idx]
            ev_values = np.arange(len(marginal))
            # Only plot non-zero range
            nonzero = marginal > 1e-6
            if np.any(nonzero):
                min_ev = np.where(nonzero)[0][0]
                max_ev = np.where(nonzero)[0][-1]
                ax.bar(
                    ev_values[min_ev:max_ev+1],
                    marginal[min_ev:max_ev+1],
                    width=1.0,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.set_xlim(min_ev - 1, max_ev + 1)
            ax.set_xlabel('EV Value')
            ax.set_ylabel('Probability')
            ax.set_title(f'{stat_names[s_idx]} EV')
        
        plt.tight_layout()
        
        # Try to save the figure
        filename = f"stat_tracker_{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        print(f"\nPlot saved to: {filename}")
        plt.close(fig)
        
    except Exception as e:
        # Fail gracefully if matplotlib is not available or other errors occur
        print(f"\nWarning: Could not create plots: {e}")


def track_training_stats(
    regimen: TrainingRegimen,
    observations: List[ObservedStats],
    base_stats: StatBlock,
    nature: Nature,
    species_info,  # SpeciesInfo for RegimenSimulator
    gen: int = 4,  # Generation for EXP calculation
    M: int = 20000,
    verbose: bool = False,
    debug_plots: bool = False,
) -> Tuple[EV_PMF, IV_PMF]:
    """
    Track Pokemon stats through a training regimen with observations.
    
    This function performs a full Bayesian IV/EV update pipeline by:
    1. Sorting and validating observations
    2. Splitting the regimen at observation levels
    3. Initializing uniform IV prior and zero-EV prior
    4. Iteratively:
       - Running RegimenSimulator over each block
       - Applying analytic_update_with_observation at observation levels
    5. Returning final posterior EV_PMF and IV_PMF
    
    Parameters
    ----------
    regimen : TrainingRegimen
        The training regimen (sequence of training blocks)
    observations : List[ObservedStats]
        List of observed stats at various levels
    base_stats : StatBlock
        Base stats for the Pokemon species
    nature : Nature
        Nature of the Pokemon
    species_info : SpeciesInfo
        Species information (for RegimenSimulator)
    gen : int
        Generation for EXP calculation (default: 4)
    M : int
        Number of Monte Carlo particles for updates (default: 20000)
    verbose : bool
        Whether to print detailed progress information (default: False)
    debug_plots : bool
        Whether to generate matplotlib plots after each observation (default: False)
        
    Returns
    -------
    Tuple[EV_PMF, IV_PMF]
        Final posterior EV and IV PMFs after all blocks and observations
        
    Raises
    ------
    ValueError
        If observation levels are outside the regimen's level range or not strictly increasing
    """
    # 1. Sort observations by level
    sorted_obs = sorted(observations, key=lambda obs: obs.level)
    
    # 2. Validate observations
    if not regimen.blocks:
        raise ValueError("Regimen has no blocks")
    
    regimen_start = regimen.blocks[0].start_level
    regimen_end = regimen.blocks[-1].end_level
    
    obs_levels = [obs.level for obs in sorted_obs]
    
    # Check all observations are within regimen range
    for obs_level in obs_levels:
        if obs_level < regimen_start or obs_level > regimen_end:
            raise ValueError(
                f"Observation level {obs_level} is outside regimen range "
                f"[{regimen_start}, {regimen_end}]"
            )
    
    # Check levels are strictly increasing (or at least non-decreasing)
    for i in range(1, len(obs_levels)):
        if obs_levels[i] < obs_levels[i-1]:
            raise ValueError(
                f"Observation levels are not sorted: {obs_levels[i]} < {obs_levels[i-1]}"
            )
    
    if verbose:
        print(f"Processing {len(sorted_obs)} observations at levels: {obs_levels}")
    
    # 3. Split regimen at observation levels
    split_regimen = split_regimen_at_levels(regimen, obs_levels)
    
    if verbose:
        print(f"Regimen split into {len(split_regimen.blocks)} blocks")
        for i, block in enumerate(split_regimen.blocks):
            print(f"  Block {i+1}: levels {block.start_level}-{block.end_level} ({block.location})")
    
    # 4. Initialize priors
    # IV prior: uniform over all 6 stats and 32 IV values
    iv_prior = IV_PMF(prior=None)  # None means uniform
    
    # EV prior: no training yet (P(T=0) = 1)
    # Create a delta distribution at T=0
    max_total_ev = 510
    priorT = np.zeros(max_total_ev + 1, dtype=float)
    priorT[0] = 1.0  # All mass at T=0
    
    # For T=0, any alpha is consistent (use uniform Dirichlet)
    alpha = np.ones(6, dtype=float)
    
    ev_prior = EV_PMF(priorT=priorT, alpha=alpha)
    
    if verbose:
        print("\nInitialized priors:")
        print("  IV: uniform distribution over [0, 31] for each stat")
        print("  EV: delta at zero (no training yet)")
    
    # 5. Create a mapping from level to observations
    obs_by_level = {}
    for obs in sorted_obs:
        if obs.level not in obs_by_level:
            obs_by_level[obs.level] = []
        obs_by_level[obs.level].append(obs)
    
    # 6. Iterate over blocks
    current_iv_pmf = iv_prior
    current_ev_pmf = ev_prior
    
    for block_idx, block in enumerate(split_regimen.blocks):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing block {block_idx+1}/{len(split_regimen.blocks)}: "
                  f"levels {block.start_level}-{block.end_level} at {block.location}")
            print(f"{'='*70}")
        
        # Create a single-block regimen for the simulator
        block_regimen = TrainingRegimen(blocks=[block])
        
        # Run RegimenSimulator over this block
        simulator = RegimenSimulator(
            regimen=block_regimen,
            species=species_info,
            gen=gen,
        )
        
        # Run simulation to get EV samples
        num_trials = M
        simulator.run_simulation(num_trials=num_trials, exp_start=species_info.exp_to_level(block.start_level))
        
        # Convert samples to EV_PMF
        post_ev_sim = simulator.toPMF(allocator="round")
        
        # Update EV prior by combining with simulation result
        from bayesian_model import update_ev_pmf
        current_ev_pmf = update_ev_pmf(current_ev_pmf, post_ev_sim, mode="linear")
        
        # IV doesn't change during simulation (no observations yet in this block)
        # Keep current_iv_pmf as is
        
        if verbose:
            print(f"\nCompleted simulation for block (levels {block.start_level}-{block.end_level})")
        
        # Check if there are observations at the end of this block
        if block.end_level in obs_by_level:
            obs_list = obs_by_level[block.end_level]
            
            if verbose:
                print(f"\nFound {len(obs_list)} observation(s) at level {block.end_level}")
            
            # Process each observation at this level
            for obs_idx, obs in enumerate(obs_list):
                if verbose:
                    print(f"\n  Applying observation {obs_idx+1}/{len(obs_list)} at level {block.end_level}")
                    print(f"  Observed stats: {obs.stats}")
                
                # Apply analytic update
                current_ev_pmf, current_iv_pmf = analytic_update_with_observation(
                    prior_ev=current_ev_pmf,
                    prior_iv=current_iv_pmf,
                    obs_stats=obs.stats,
                    level=obs.level,
                    base_stats=base_stats,
                    nature=nature,
                    M=M,
                    verbose=verbose,
                )
                
                if verbose:
                    print(f"  Completed Bayesian update for observation at level {block.end_level}")
                
                # Optional: plot after each observation
                if debug_plots:
                    plot_marginals(
                        current_ev_pmf,
                        current_iv_pmf,
                        title=f"After Observation at Level {block.end_level}",
                    )
    
    # 7. Print final histograms
    print_iv_histograms(current_iv_pmf, title="Final IV Marginal Distributions")
    print_ev_histograms(current_ev_pmf, title="Final EV Marginal Distributions")
    
    # 8. Optional: final debug plot
    if debug_plots:
        plot_marginals(
            current_ev_pmf,
            current_iv_pmf,
            title="Final Posteriors",
        )
    
    return current_ev_pmf, current_iv_pmf
