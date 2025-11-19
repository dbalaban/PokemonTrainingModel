#!/usr/bin/env python3
"""
Hyperparameter search for the Bayesian update methods.

This script performs a systematic search over:
1. Update method (analytic_update_with_observation, hybrid_ev_iv_update, update_with_observation)
2. Smoothing parameters (EV PMF smoothing to handle data inconsistencies)
3. Batch size and max_batches parameters
4. Tolerance parameters

The goal is to find parameter combinations that allow successful sample selection
in the update step, particularly for the problematic level 20 observation.
"""

import sys
sys.path.insert(0, '../src')

from data_structures import *
from PMFs import EV_PMF, IV_PMF
from regimen_sim import RegimenSimulator
from bayesian_model import (
    analytic_update_with_observation,
    hybrid_ev_iv_update, 
    update_with_observation
)
import numpy as np
import argparse
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

# Species definitions (same as run_regimen.py)
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
    TrainingBlock(location="Route 201", start_level=1, end_level=5,
                  encounters=[EncounterOption(target=Starly, weight=1.0, levels=[2,3])]),
    TrainingBlock(location="Route 202", start_level=5, end_level=8,
                  encounters=[EncounterOption(target=Shinx, weight=1.0, levels=[3,4])]),
    TrainingBlock(location="Route 203", start_level=8, end_level=9,
                  encounters=[EncounterOption(target=Starly, weight=0.25, levels=[4,6,7]),
                             EncounterOption(target=Starly, weight=0.1, levels=[5])]),
    TrainingBlock(location="Route 203", start_level=9, end_level=10,
                  encounters=[EncounterOption(target=Shinx, weight=1.0, levels=[4,5])]),
    TrainingBlock(location="Oreburgh Gate", start_level=10, end_level=12,
                  encounters=[EncounterOption(target=Zubat, weight=1.0, levels=[5,6,7,8])]),
    TrainingBlock(location="Route 207", start_level=12, end_level=13,
                  encounters=[EncounterOption(target=Machop, weight=1.0, levels=[5,6,7,8])]),
    TrainingBlock(location="Route 207", start_level=13, end_level=18,
                  encounters=[EncounterOption(target=Ponyta, weight=1.0, levels=[5,6,7])]),
    TrainingBlock(location="Route 207", start_level=18, end_level=20,
                  encounters=[EncounterOption(target=Machop, weight=1.0, levels=[5,6,7,8])]),
])


def apply_ev_smoothing(ev_pmf: EV_PMF, smoothing_alpha: float, smoothing_T: float) -> EV_PMF:
    """
    Apply smoothing to an EV_PMF by mixing with a wider distribution.
    
    Parameters
    ----------
    ev_pmf : EV_PMF
        The original EV PMF
    smoothing_alpha : float
        Smoothing strength for the Dirichlet concentration parameter (0 = no smoothing, 1 = full smoothing)
        Higher values make the proportions more uniform
    smoothing_T : float
        Smoothing strength for the total EV distribution (0 = no smoothing, 1 = full smoothing)
        Higher values widen the distribution
        
    Returns
    -------
    EV_PMF
        Smoothed EV PMF
    """
    if smoothing_alpha == 0.0 and smoothing_T == 0.0:
        return ev_pmf
    
    # Smooth alpha (concentration parameters)
    if smoothing_alpha > 0:
        # Mix current alpha with uniform concentration
        uniform_alpha = np.ones(6, dtype=float)
        new_alpha = (1 - smoothing_alpha) * ev_pmf.alpha + smoothing_alpha * uniform_alpha
    else:
        new_alpha = ev_pmf.alpha.copy()
    
    # Smooth T (total EV distribution)
    if smoothing_T > 0:
        # Create a wider distribution by convolving with a small uniform
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
    
    return EV_PMF(priorT=new_T, alpha=new_alpha, rng=ev_pmf.rng)


def test_update_configuration(
    config: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test a specific configuration of hyperparameters.
    
    Parameters
    ----------
    config : dict
        Configuration containing:
        - method: str - 'analytic', 'hybrid', or 'simple'
        - M: int - number of Monte Carlo particles
        - smoothing_alpha: float - alpha smoothing parameter (0-1)
        - smoothing_T: float - T smoothing parameter (0-1)
        - batch_size: int (for analytic method)
        - max_batches: int (for analytic method)
        - tol: int - tolerance (for simple/hybrid methods)
        - max_iters: int (for hybrid method)
        
    Returns
    -------
    dict
        Results containing:
        - success: bool - whether the update succeeded
        - valid_samples: int - number of valid samples found
        - error: str - error message if failed
    """
    method = config['method']
    M = config['M']
    smoothing_alpha = config.get('smoothing_alpha', 0.0)
    smoothing_T = config.get('smoothing_T', 0.0)
    
    if verbose:
        print(f"\nTesting configuration:")
        print(f"  Method: {method}")
        print(f"  M: {M}")
        print(f"  Smoothing alpha: {smoothing_alpha}")
        print(f"  Smoothing T: {smoothing_T}")
    
    try:
        # Simulate up to level 19 to get the priors
        from stat_tracker import track_training_stats, split_regimen_at_levels
        
        observations_19 = [
            ObservedStats(level=12, stats=StatBlock(hp=32, atk=24, def_=15, spa=16, spd=15, spe=23)),
            ObservedStats(level=19, stats=StatBlock(hp=46, atk=35, def_=22, spa=23, spd=21, spe=35)),
        ]
        
        nature = Nature(name="Hardy", inc=None, dec=None)
        
        # Run up to level 19
        ev_pmf_19, iv_pmf_19 = track_training_stats(
            regimen=regimen,
            observations=observations_19,
            base_stats=Riolu.base_stats,
            nature=nature,
            species_info=Riolu,
            gen=4,
            M=M,
            verbose=False,
            debug_plots=False,
        )
        
        # Apply smoothing
        if smoothing_alpha > 0 or smoothing_T > 0:
            ev_pmf_19 = apply_ev_smoothing(ev_pmf_19, smoothing_alpha, smoothing_T)
            if verbose:
                print(f"  Applied smoothing to EV PMF")
        
        # Now try to update with level 20 observation
        obs_20 = ObservedStats(level=20, stats=StatBlock(hp=48, atk=37, def_=23, spa=24, spd=22, spe=36))
        
        # First, simulate from level 19 to 20
        blocks_19_20 = [b for b in regimen.blocks if b.start_level >= 18 and b.end_level <= 20]
        if blocks_19_20:
            regimen_19_20 = TrainingRegimen(blocks=blocks_19_20)
            simulator = RegimenSimulator(regimen=regimen_19_20, species=Riolu, gen=4)
            simulator.run_simulation(num_trials=M, exp_start=Riolu.exp_to_level(19))
            post_ev_sim = simulator.toPMF(allocator="round")
            
            from bayesian_model import update_ev_pmf
            ev_pmf_20_pre = update_ev_pmf(ev_pmf_19, post_ev_sim, mode="linear")
        else:
            ev_pmf_20_pre = ev_pmf_19
        
        # Apply smoothing again before the final update
        if smoothing_alpha > 0 or smoothing_T > 0:
            ev_pmf_20_pre = apply_ev_smoothing(ev_pmf_20_pre, smoothing_alpha, smoothing_T)
        
        # Try the update
        if method == 'analytic':
            batch_size = config.get('batch_size', M)
            max_batches = config.get('max_batches', 100)
            
            ev_pmf_20, iv_pmf_20 = analytic_update_with_observation(
                prior_ev=ev_pmf_20_pre,
                prior_iv=iv_pmf_19,
                obs_stats=obs_20.stats,
                level=obs_20.level,
                base_stats=Riolu.base_stats,
                nature=nature,
                M=M,
                verbose=verbose,
                batch_size=batch_size,
                max_batches=max_batches,
            )
            
            # Check if we got valid samples (heuristic: if returned same as input, failed)
            success = not np.allclose(ev_pmf_20.T, ev_pmf_20_pre.T)
            
        elif method == 'hybrid':
            tol = config.get('tol', 0)
            max_iters = config.get('max_iters', 5)
            
            ev_pmf_20, iv_pmf_20 = hybrid_ev_iv_update(
                prior_ev=ev_pmf_20_pre,
                prior_iv=iv_pmf_19,
                obs_stats=obs_20.stats,
                level=obs_20.level,
                base_stats=Riolu.base_stats,
                nature=nature,
                mc_particles=M,
                tol=tol,
                max_iters=max_iters,
                verbose=verbose,
            )
            success = True  # hybrid uses soft fallback
            
        elif method == 'simple':
            tol = config.get('tol', 0)
            
            ev_pmf_20, iv_pmf_20 = update_with_observation(
                prior_ev=ev_pmf_20_pre,
                prior_iv=iv_pmf_19,
                obs_stats=obs_20.stats,
                level=obs_20.level,
                base_stats=Riolu.base_stats,
                nature=nature,
                M=M,
                tol=tol,
                verbose=verbose,
            )
            success = True  # simple uses soft fallback
        else:
            return {'success': False, 'valid_samples': 0, 'error': f'Unknown method: {method}'}
        
        return {
            'success': success,
            'valid_samples': M if success else 0,
            'error': None,
            'config': config
        }
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return {
            'success': False,
            'valid_samples': 0,
            'error': str(e),
            'config': config
        }


def hyperparameter_search(
    methods: List[str] = ['analytic', 'hybrid', 'simple'],
    M_values: List[int] = [1000, 5000],
    smoothing_alpha_values: List[float] = [0.0, 0.1, 0.3, 0.5],
    smoothing_T_values: List[float] = [0.0, 0.1, 0.3, 0.5],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Perform systematic hyperparameter search.
    
    Returns
    -------
    list of dict
        Results for each configuration tested
    """
    results = []
    
    total_configs = len(methods) * len(M_values) * len(smoothing_alpha_values) * len(smoothing_T_values)
    config_num = 0
    
    for method in methods:
        for M in M_values:
            for smoothing_alpha in smoothing_alpha_values:
                for smoothing_T in smoothing_T_values:
                    config_num += 1
                    
                    config = {
                        'method': method,
                        'M': M,
                        'smoothing_alpha': smoothing_alpha,
                        'smoothing_T': smoothing_T,
                    }
                    
                    # Method-specific parameters
                    if method == 'analytic':
                        config['batch_size'] = M
                        config['max_batches'] = 100
                    elif method == 'hybrid':
                        config['tol'] = 0
                        config['max_iters'] = 5
                    elif method == 'simple':
                        config['tol'] = 0
                    
                    print(f"\n{'='*70}")
                    print(f"Configuration {config_num}/{total_configs}")
                    print(f"{'='*70}")
                    
                    result = test_update_configuration(config, verbose=verbose)
                    results.append(result)
                    
                    if result['success']:
                        print(f"✓ SUCCESS - Found {result['valid_samples']} valid samples")
                    else:
                        print(f"✗ FAILED - {result.get('error', 'No valid samples')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for Bayesian update methods')
    parser.add_argument('--methods', nargs='+', default=['analytic', 'hybrid', 'simple'],
                        choices=['analytic', 'hybrid', 'simple'],
                        help='Update methods to test')
    parser.add_argument('--M', nargs='+', type=int, default=[1000, 5000],
                        help='Monte Carlo particle counts to test')
    parser.add_argument('--smoothing-alpha', nargs='+', type=float, default=[0.0, 0.1, 0.3, 0.5],
                        help='Alpha smoothing values to test')
    parser.add_argument('--smoothing-T', nargs='+', type=float, default=[0.0, 0.1, 0.3, 0.5],
                        help='T smoothing values to test')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HYPERPARAMETER SEARCH FOR BAYESIAN UPDATE METHODS")
    print("="*70)
    print(f"Methods: {args.methods}")
    print(f"M values: {args.M}")
    print(f"Smoothing alpha values: {args.smoothing_alpha}")
    print(f"Smoothing T values: {args.smoothing_T}")
    
    results = hyperparameter_search(
        methods=args.methods,
        M_values=args.M,
        smoothing_alpha_values=args.smoothing_alpha,
        smoothing_T_values=args.smoothing_T,
        verbose=args.verbose
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ SUCCESSFUL CONFIGURATIONS:")
        for r in successful:
            cfg = r['config']
            print(f"  - Method: {cfg['method']}, M: {cfg['M']}, "
                  f"smoothing_alpha: {cfg['smoothing_alpha']}, smoothing_T: {cfg['smoothing_T']}")
    
    if failed:
        print(f"\n✗ FAILED CONFIGURATIONS:")
        for r in failed[:5]:  # Show first 5
            cfg = r['config']
            print(f"  - Method: {cfg['method']}, M: {cfg['M']}, "
                  f"smoothing_alpha: {cfg['smoothing_alpha']}, smoothing_T: {cfg['smoothing_T']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'summary': {
                    'total': len(results),
                    'successful': len(successful),
                    'failed': len(failed)
                }
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
