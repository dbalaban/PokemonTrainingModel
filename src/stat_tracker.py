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


def apply_ev_smoothing(ev_pmf: EV_PMF, smoothing_alpha: float = 0.0, smoothing_T: float = 0.0) -> EV_PMF:
    """
    Apply smoothing to an EV_PMF by mixing with a wider distribution.
    
    This function is mode-agnostic and handles both 'dirichlet' and 'histogram' modes:
      - Dirichlet mode: Smooths alpha and T parameters
      - Histogram mode: Smooths independent histograms via Gaussian blur
    
    This function can help handle minor data inconsistencies by widening
    the EV distribution, making it less concentrated and more robust to
    observation noise.
    
    Parameters
    ----------
    ev_pmf : EV_PMF
        The original EV PMF
    smoothing_alpha : float, optional
        Smoothing strength parameter (default: 0.0)
        - In dirichlet mode: smooths Dirichlet concentration parameter
        - In histogram mode: controls Gaussian blur width
        - 0.0 = no smoothing (use original)
        - 1.0 = full smoothing
        Higher values make the distributions wider/less concentrated
    smoothing_T : float, optional
        Smoothing strength for the total EV distribution (default: 0.0)
        - In dirichlet mode: smooths total EV distribution
        - In histogram mode: additional global smoothing
        - 0.0 = no smoothing (use original)
        - 1.0 = full smoothing (wide uniform)
        Higher values widen the total EV distribution
        
    Returns
    -------
    EV_PMF
        Smoothed EV PMF with the same mode as input
        
    Notes
    -----
    Smoothing is useful when:
    - Observations have minor measurement errors
    - Training regimen is approximate (not all encounters recorded)
    - Some flexibility is needed in the inference
    
    However, smoothing cannot fix fundamental data inconsistencies where
    observed stats are incompatible with the training regimen.
    """
    if smoothing_alpha == 0.0 and smoothing_T == 0.0:
        return ev_pmf
    
    if ev_pmf.mode == 'dirichlet':
        # ---- Dirichlet mode: smooth alpha and T ----
        
        # Smooth alpha (concentration parameters)
        if smoothing_alpha > 0:
            # Mix current alpha with uniform concentration
            uniform_alpha = np.ones(6, dtype=float)
            new_alpha = (1 - smoothing_alpha) * ev_pmf.alpha + smoothing_alpha * uniform_alpha
        else:
            new_alpha = ev_pmf.alpha.copy()
        
        # Smooth T (total EV distribution)
        if smoothing_T > 0:
            # Create a wider distribution by mixing with a uniform around the expected total
            current_T = ev_pmf.T.copy()
            # Expected total
            exp_total = float(np.dot(np.arange(len(current_T)), current_T))
            # Create a uniform distribution around the expected total
            width = int(smoothing_T * 50)  # Width proportional to smoothing
            if width > 0:
                uniform_T = np.zeros_like(current_T)
                lo = max(0, int(exp_total - width))
                hi = min(len(current_T) - 1, int(exp_total + width))
                uniform_T[lo:hi+1] = 1.0 / (hi - lo + 1)
                # Mix
                new_T = (1 - smoothing_T) * current_T + smoothing_T * uniform_T
                new_T = new_T / new_T.sum()  # Renormalize
            else:
                new_T = current_T
        else:
            new_T = ev_pmf.T.copy()
        
        return EV_PMF(priorT=new_T, alpha=new_alpha, mode='dirichlet', rng=ev_pmf.rng)
    
    elif ev_pmf.mode == 'histogram':
        # ---- Histogram mode: apply Gaussian smoothing to histograms ----
        
        if smoothing_alpha == 0.0 and smoothing_T == 0.0:
            return ev_pmf
        
        # Combine smoothing parameters for histogram mode
        smoothing_strength = max(smoothing_alpha, smoothing_T)
        
        if smoothing_strength == 0.0:
            return ev_pmf
        
        # Apply Gaussian blur to each histogram
        max_ev = ev_pmf.max_ev
        new_histograms = np.zeros((6, max_ev + 1), dtype=float)
        
        # Gaussian kernel width based on smoothing strength
        sigma = smoothing_strength * 10.0  # Scale to reasonable width
        
        if sigma < 0.1:
            # Too small to matter, return original
            return ev_pmf
        
        # Create Gaussian kernel
        kernel_width = int(np.ceil(3 * sigma))
        kernel_size = 2 * kernel_width + 1
        x = np.arange(-kernel_width, kernel_width + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize
        
        for s in range(6):
            # Convolve with Gaussian kernel
            smoothed = np.convolve(ev_pmf.histograms[s], kernel, mode='same')
            
            # Ensure non-negative and normalize
            smoothed = np.maximum(smoothed, 0.0)
            total = smoothed.sum()
            if total > 0:
                new_histograms[s] = smoothed / total
            else:
                new_histograms[s] = ev_pmf.histograms[s].copy()
        
        return EV_PMF(mode='histogram', histograms=new_histograms, rng=ev_pmf.rng)
    
    else:
        raise ValueError(f"Unknown EV_PMF mode: {ev_pmf.mode}")


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
    output_dir: str = "plots",
    counter: int = 0,
    plot_ev: bool = True,
    plot_iv: bool = True,
) -> None:
    """
    Plot IV and EV marginal distributions on single plots with color-coded lines.
    
    This function creates two separate plots:
    - One for IV distributions (all stats on one plot with legend) - if plot_iv is True
    - One for EV distributions (all stats on one plot with legend) - if plot_ev is True
    
    Stats where P(x=0)=1 are not plotted (assumed to be always 0).
    
    Parameters
    ----------
    ev_pmf : EV_PMF
        The EV PMF to plot
    iv_pmf : IV_PMF
        The IV PMF to plot
    title : str
        Title for the plot
    output_dir : str
        Directory to save the plots (default: "plots")
    counter : int
        Counter for ordering the plots in file names (default: 0)
    plot_ev : bool
        Whether to plot EV distributions (default: True)
    plot_iv : bool
        Whether to plot IV distributions (default: True)
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        stat_names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Sanitize title for filename
        safe_title = title.replace(' ', '_').replace('/', '_').lower()
        
        # ==================== Plot IV marginals ====================
        if plot_iv:
            fig_iv, ax_iv = plt.subplots(figsize=(10, 6))
            iv_values = np.arange(32)
            
            # Track which stats to plot
            plotted_any_iv = False
            
            for s_idx in range(6):
                marginal = iv_pmf.P[s_idx]
                
                # Check if P(x=0) = 1, which means stat is always 0
                if marginal[0] > 0.9999:
                    # Skip plotting this stat
                    continue
                
                # Plot this stat
                ax_iv.plot(iv_values, marginal, label=stat_names[s_idx], 
                          color=colors[s_idx], linewidth=2, marker='o', markersize=3)
                plotted_any_iv = True
            
            # Configure IV plot
            ax_iv.set_xlim(0, 31)
            ax_iv.set_ylim(0, 1)
            ax_iv.set_xlabel('IV Value', fontsize=12)
            ax_iv.set_ylabel('Probability', fontsize=12)
            ax_iv.set_title(f'IV Distributions - {title}', fontsize=14)
            ax_iv.grid(True, alpha=0.3)
            
            if plotted_any_iv:
                ax_iv.legend(loc='best', fontsize=10)
            
            # Save IV plot
            iv_filename = os.path.join(output_dir, f"{counter:03d}_iv_{safe_title}.png")
            plt.tight_layout()
            fig_iv.savefig(iv_filename, dpi=100)
            print(f"IV plot saved to: {iv_filename}")
            plt.close(fig_iv)
        
        # ==================== Plot EV marginals ====================
        if plot_ev:
            fig_ev, ax_ev = plt.subplots(figsize=(10, 6))
            
            # Get EV marginals
            ev_marginals = ev_pmf.getMarginals(mc_samples=5000)
            
            # Determine max EV value (252 per stat)
            max_ev_value = 252
            ev_values = np.arange(max_ev_value + 1)
            
            # Track which stats to plot
            plotted_any_ev = False
            
            for s_idx in range(6):
                marginal = ev_marginals[s_idx]
                
                # Check if P(x=0) = 1, which means stat is always 0
                if len(marginal) > 0 and marginal[0] > 0.9999:
                    # Skip plotting this stat
                    continue
                
                # Pad marginal if needed
                if len(marginal) <= max_ev_value:
                    padded_marginal = np.zeros(max_ev_value + 1)
                    padded_marginal[:len(marginal)] = marginal
                else:
                    padded_marginal = marginal[:max_ev_value + 1]
                
                # Plot this stat
                ax_ev.plot(ev_values, padded_marginal, label=stat_names[s_idx], 
                          color=colors[s_idx], linewidth=2, marker='o', markersize=2, 
                          markevery=max(1, max_ev_value // 20))
                plotted_any_ev = True
            
            # Configure EV plot
            ax_ev.set_xlim(0, max_ev_value)
            ax_ev.set_ylim(0, 1)
            ax_ev.set_xlabel('EV Value', fontsize=12)
            ax_ev.set_ylabel('Probability', fontsize=12)
            ax_ev.set_title(f'EV Distributions - {title}', fontsize=14)
            ax_ev.grid(True, alpha=0.3)
            
            if plotted_any_ev:
                ax_ev.legend(loc='best', fontsize=10)
            
            # Save EV plot
            ev_filename = os.path.join(output_dir, f"{counter:03d}_ev_{safe_title}.png")
            plt.tight_layout()
            fig_ev.savefig(ev_filename, dpi=100)
            print(f"EV plot saved to: {ev_filename}")
            plt.close(fig_ev)
        
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
    smoothing_alpha: float = 0.0,
    smoothing_T: float = 0.0,
    update_method: str = 'analytic',
) -> Tuple[EV_PMF, IV_PMF]:
    """
    Track Pokemon stats through a training regimen with observations.
    
    This function performs a full Bayesian IV/EV update pipeline by:
    1. Sorting and validating observations
    2. Splitting the regimen at observation levels
    3. Initializing uniform IV prior and zero-EV prior
    4. Iteratively:
       - Running RegimenSimulator over each block
       - Applying Bayesian update at observation levels
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
    smoothing_alpha : float
        EV PMF alpha smoothing parameter (0.0=no smoothing, 1.0=full) (default: 0.0)
    smoothing_T : float
        EV PMF total smoothing parameter (0.0=no smoothing, 1.0=full) (default: 0.0)
    update_method : str
        Update method to use: 'analytic', 'hybrid', or 'simple' (default: 'analytic')
        
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
    
    # 6. Group blocks into regimens between observations
    # Each regimen runs from start (or previous observation) to next observation
    regimen_groups = []
    current_group = []
    
    for block in split_regimen.blocks:
        current_group.append(block)
        # If this block ends at an observation level, close the group
        if block.end_level in obs_by_level:
            regimen_groups.append(current_group)
            current_group = []
    
    # Add any remaining blocks (after the last observation)
    if current_group:
        regimen_groups.append(current_group)
    
    if verbose:
        print(f"\nGrouped into {len(regimen_groups)} regimen(s) between observations:")
        for i, group in enumerate(regimen_groups):
            start_level = group[0].start_level
            end_level = group[-1].end_level
            print(f"  Regimen {i+1}: levels {start_level}-{end_level} ({len(group)} block(s))")
    
    # 7. Iterate over regimen groups
    current_iv_pmf = iv_prior
    current_ev_pmf = ev_prior
    
    # Counter for ordering debug plots
    plot_counter = 0
    
    for regimen_idx, blocks_group in enumerate(regimen_groups):
        # Get the start and end levels for this regimen group
        regimen_start_level = blocks_group[0].start_level
        regimen_end_level = blocks_group[-1].end_level
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing regimen {regimen_idx+1}/{len(regimen_groups)}: "
                  f"levels {regimen_start_level}-{regimen_end_level}")
            print(f"  Contains {len(blocks_group)} block(s)")
            print(f"{'='*70}")
        
        # Create a regimen with all blocks in this group
        group_regimen = TrainingRegimen(blocks=blocks_group)
        
        # Run RegimenSimulator over the entire regimen group (cumulative)
        simulator = RegimenSimulator(
            regimen=group_regimen,
            species=species_info,
            gen=gen,
        )
        
        # Run simulation to get EV samples for the entire regimen
        num_trials = M
        simulator.run_simulation(num_trials=num_trials, exp_start=species_info.exp_to_level(regimen_start_level))
        
        # Convert samples to EV_PMF (match mode of current EV PMF)
        post_ev_sim = simulator.toPMF(allocator="round", mode=current_ev_pmf.mode)
        
        # Verbose: Show EV statistics before and after combining
        if verbose:
            stat_names = ['HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed']
            
            # Get marginals for current EV PMF (before adding simulation)
            current_ev_marginals = current_ev_pmf.getMarginals(mc_samples=10000)  # (6, 253)
            current_ev_means = np.array([
                np.dot(np.arange(current_ev_marginals.shape[1]), current_ev_marginals[s])
                for s in range(6)
            ])
            
            # Get marginals for simulated EV gains
            sim_ev_marginals = post_ev_sim.getMarginals(mc_samples=10000)  # (6, 253)
            sim_ev_means = np.array([
                np.dot(np.arange(sim_ev_marginals.shape[1]), sim_ev_marginals[s])
                for s in range(6)
            ])
            
            print(f"\n  EV Statistics for Regimen {regimen_idx+1} (levels {regimen_start_level}-{regimen_end_level}):")
            print(f"  {'Stat':<12} | {'Current EV':>12} | {'Simulated Gain':>15} | {'Expected Sum':>13}")
            print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*13}")
            for s in range(6):
                # Only show non-zero values
                if current_ev_means[s] > 0.01 or sim_ev_means[s] > 0.01:
                    expected_sum = current_ev_means[s] + sim_ev_means[s]
                    print(f"  {stat_names[s]:<12} | {current_ev_means[s]:>12.2f} | {sim_ev_means[s]:>15.2f} | {expected_sum:>13.2f}")
        
        # Optional: plot EV gains from simulation (EV only, no IV change yet)
        if debug_plots:
            plot_marginals(
                post_ev_sim,
                current_iv_pmf,
                title=f"EV Gains from Simulation Regimen {regimen_idx+1} Levels {regimen_start_level}-{regimen_end_level}",
                output_dir="plots/simulated_ev",
                counter=plot_counter,
                plot_ev=True,
                plot_iv=False,  # IV doesn't change during simulation
            )
            plot_counter += 1
        
        # Update EV prior by combining with simulation result
        from bayesian_model import update_ev_pmf
        current_ev_pmf = update_ev_pmf(current_ev_pmf, post_ev_sim, mode="linear")
        
        # Verbose: Show actual EV after combining (verify additivity)
        if verbose:
            # Get marginals for updated EV PMF (after combination)
            updated_ev_marginals = current_ev_pmf.getMarginals(mc_samples=10000)  # (6, 253)
            updated_ev_means = np.array([
                np.dot(np.arange(updated_ev_marginals.shape[1]), updated_ev_marginals[s])
                for s in range(6)
            ])
            
            print(f"  {'Stat':<12} | {'Actual After':>13} | {'Difference':>11}")
            print(f"  {'-'*12}-+-{'-'*13}-+-{'-'*11}")
            for s in range(6):
                # Only show non-zero values
                if current_ev_means[s] > 0.01 or sim_ev_means[s] > 0.01:
                    expected_sum = current_ev_means[s] + sim_ev_means[s]
                    difference = updated_ev_means[s] - expected_sum
                    print(f"  {stat_names[s]:<12} | {updated_ev_means[s]:>13.2f} | {difference:>11.2f}")
            print()
        
        # Optional: plot updated EV PMF after combining with prior (EV only, IV still unchanged)
        if debug_plots:
            plot_marginals(
                current_ev_pmf,
                current_iv_pmf,
                title=f"Updated EV PMF Regimen {regimen_idx+1} Levels {regimen_start_level}-{regimen_end_level}",
                output_dir="plots/updated_pmf",
                counter=plot_counter,
                plot_ev=True,
                plot_iv=False,  # IV still doesn't change until observation
            )
            plot_counter += 1
        
        # IV doesn't change during simulation (no observations yet)
        # Keep current_iv_pmf as is
        
        if verbose:
            print(f"\nCompleted simulation for regimen (levels {regimen_start_level}-{regimen_end_level})")
        
        # Check if there are observations at the end of this regimen
        if regimen_end_level in obs_by_level:
            obs_list = obs_by_level[regimen_end_level]
            
            if verbose:
                print(f"\nFound {len(obs_list)} observation(s) at level {regimen_end_level}")
            
            # Process each observation at this level
            for obs_idx, obs in enumerate(obs_list):
                if verbose:
                    print(f"\n  Applying observation {obs_idx+1}/{len(obs_list)} at level {regimen_end_level}")
                    print(f"  Observed stats: {obs.stats}")
                    if smoothing_alpha > 0 or smoothing_T > 0:
                        print(f"  Smoothing parameters: alpha={smoothing_alpha}, T={smoothing_T}")
                    print(f"  Update method: {update_method}")
                
                # Apply smoothing before update if requested
                if smoothing_alpha > 0 or smoothing_T > 0:
                    current_ev_pmf = apply_ev_smoothing(current_ev_pmf, smoothing_alpha, smoothing_T)
                    if verbose:
                        print(f"  Applied EV smoothing")
                
                # Apply update based on selected method
                if update_method == 'analytic':
                    from bayesian_model import analytic_update_with_observation
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
                elif update_method == 'hybrid':
                    from bayesian_model import hybrid_ev_iv_update
                    current_ev_pmf, current_iv_pmf = hybrid_ev_iv_update(
                        prior_ev=current_ev_pmf,
                        prior_iv=current_iv_pmf,
                        obs_stats=obs.stats,
                        level=obs.level,
                        base_stats=base_stats,
                        nature=nature,
                        mc_particles=M,
                        tol=0,
                        max_iters=5,
                        verbose=verbose,
                    )
                elif update_method == 'simple':
                    from bayesian_model import update_with_observation
                    current_ev_pmf, current_iv_pmf = update_with_observation(
                        prior_ev=current_ev_pmf,
                        prior_iv=current_iv_pmf,
                        obs_stats=obs.stats,
                        level=obs.level,
                        base_stats=base_stats,
                        nature=nature,
                        M=M,
                        tol=0,
                        verbose=verbose,
                    )
                else:
                    raise ValueError(f"Unknown update method: {update_method}")
                
                if verbose:
                    print(f"  Completed Bayesian update for observation at level {regimen_end_level}")
                
                # Optional: plot both IV and EV after observation (both change now)
                if debug_plots:
                    plot_marginals(
                        current_ev_pmf,
                        current_iv_pmf,
                        title=f"After Observation Level {regimen_end_level} Obs {obs_idx+1}",
                        output_dir="plots/observation_update",
                        counter=plot_counter,
                        plot_ev=True,
                        plot_iv=True,  # Both IV and EV updated by observation
                    )
                    plot_counter += 1
    
    # 7. Print final histograms
    print_iv_histograms(current_iv_pmf, title="Final IV Marginal Distributions")
    print_ev_histograms(current_ev_pmf, title="Final EV Marginal Distributions")
    
    # 8. Optional: final debug plot (both IV and EV)
    if debug_plots:
        plot_marginals(
            current_ev_pmf,
            current_iv_pmf,
            title="Final Posteriors",
            output_dir="plots",
            counter=plot_counter,
            plot_ev=True,
            plot_iv=True,
        )
    
    return current_ev_pmf, current_iv_pmf
