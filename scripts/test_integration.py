#!/usr/bin/env python3
"""
Comprehensive integration test verifying all fixes work together.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from data_structures import StatBlock, StatType, statblock_to_array, array_to_statblock
from PMFs import EV_PMF, IV_PMF

def test_all_fixes():
    """Integration test covering all the implemented fixes."""
    print("=" * 60)
    print("COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: StatBlock indexing with all stats
    print("\n1. Testing StatBlock indexing fix...")
    sb = StatBlock(hp=100, atk=120, def_=80, spa=90, spd=70, spe=110)
    assert sb[StatType.HP] == 100
    assert sb[StatType.ATTACK] == 120
    assert sb[StatType.DEFENSE] == 80
    assert sb[StatType.SPECIAL_ATTACK] == 90
    assert sb[StatType.SPECIAL_DEFENSE] == 70
    assert sb[StatType.SPEED] == 110
    sb[StatType.ATTACK] = 125
    assert sb.atk == 125
    print("   ✓ StatBlock indexing works for all 6 stats")
    
    # Test 2: StatBlock helpers
    print("\n2. Testing centralized StatBlock helpers...")
    arr = statblock_to_array(sb)
    assert arr.shape == (6,)
    assert arr[1] == 125  # atk
    sb2 = array_to_statblock(arr)
    assert sb2.atk == 125
    print("   ✓ StatBlock ↔ array conversion works")
    
    # Test 3: IV_PMF.getProb consistency
    print("\n3. Testing IV_PMF.getProb fix...")
    rng = np.random.default_rng(42)
    iv_pmf = IV_PMF(rng=rng)
    
    # Single IV
    iv_single = np.array([15, 20, 10, 31, 0, 25])
    prob_single = iv_pmf.getProb(iv_single)
    logprob_single = iv_pmf.getLogProb(iv_single)
    assert np.isclose(prob_single, np.exp(logprob_single))
    
    # Batch IVs
    iv_batch = rng.integers(0, 32, size=(6, 5))
    prob_batch = iv_pmf.getProb(iv_batch)
    logprob_batch = iv_pmf.getLogProb(iv_batch)
    assert np.allclose(prob_batch, np.exp(logprob_batch))
    print("   ✓ IV_PMF.getProb is consistent with getLogProb")
    
    # Test 4: EV_PMF.MAX_EV constant
    print("\n4. Testing EV_PMF.MAX_EV class constant...")
    assert EV_PMF.MAX_EV == 252
    ev_pmf1 = EV_PMF()
    ev_pmf2 = EV_PMF()
    assert ev_pmf1.max_ev == ev_pmf2.max_ev == 252
    assert ev_pmf1.max_total_ev == 510
    try:
        ev_pmf1.max_ev = 100
        assert False, "Should not be able to set max_ev"
    except AttributeError:
        pass
    print("   ✓ EV_PMF.MAX_EV is a shared constant with read-only property")
    
    # Test 5: Optional smoothing
    print("\n5. Testing EV_PMF optional smoothing...")
    ev_samples = np.array([
        [100, 100, 100, 100, 100, 10],
        [100, 100, 100, 100, 100, 10],
    ], dtype=float)
    
    ev_pmf_strict = EV_PMF.from_samples(ev_samples, w_bins=50, smooth_W=False)
    ev_pmf_smooth = EV_PMF.from_samples(ev_samples, w_bins=50, smooth_W=True, smooth_eps=1e-6)
    
    zeros_strict = (ev_pmf_strict.W == 0).sum()
    zeros_smooth = (ev_pmf_smooth.W == 0).sum()
    assert zeros_smooth <= zeros_strict
    print(f"   ✓ Smoothing reduces zeros: {zeros_strict} → {zeros_smooth}")
    
    # Test 6: Per-stat cap enforcement
    print("\n6. Testing per-stat EV cap enforcement...")
    # Create EV that exceeds per-stat cap
    ev_invalid = np.array([300, 100, 100, 50, 50, 50])  # First stat exceeds 252
    prob_invalid = ev_pmf_strict.getProb(ev_invalid)
    assert prob_invalid == 0.0, "Invalid EV should have zero probability"
    print("   ✓ EVs exceeding max_ev=252 are treated as infeasible")
    
    # Test 7: Total cap enforcement
    print("\n7. Testing total EV cap enforcement...")
    # Create EV that exceeds total cap
    ev_over_total = np.array([252, 252, 7, 0, 0, 0])  # Total = 511 > 510
    prob_over_total = ev_pmf_strict.getProb(ev_over_total)
    assert prob_over_total == 0.0, "EV exceeding total cap should have zero probability"
    print("   ✓ EVs exceeding max_total_ev=510 are treated as infeasible")
    
    # Test 8: Verbose flag (just check it doesn't crash)
    print("\n8. Testing verbose flags...")
    from bayesian_model import analytic_update_with_observation
    from data_structures import Nature
    
    # This should run without printing
    base = StatBlock(80, 95, 85, 65, 65, 90)
    obs = StatBlock(156, 133, 107, 90, 88, 111)
    nature = Nature("neutral", None, None)
    
    # Just verify it runs with verbose=False (no output expected)
    _, _ = analytic_update_with_observation(
        EV_PMF(), IV_PMF(), obs, 50, base, nature, M=100, verbose=False
    )
    print("   ✓ Verbose flag works (no debug output when False)")
    
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_all_fixes()
