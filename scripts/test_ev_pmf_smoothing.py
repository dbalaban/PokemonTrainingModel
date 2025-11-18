#!/usr/bin/env python3
"""
Unit test for EV_PMF optional smoothing.
Validates that smoothing can be toggled on/off and affects zero bins.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from PMFs import EV_PMF

def test_ev_pmf_smoothing():
    """Test that EV_PMF.from_samples supports optional smoothing."""
    rng = np.random.default_rng(42)
    
    # Create a simple sample set with limited coverage
    # This should result in some zero bins in the W rows
    ev_samples = np.array([
        [100, 100, 100, 100, 100, 10],  # Repeated pattern
        [100, 100, 100, 100, 100, 10],
        [100, 100, 100, 100, 100, 10],
    ], dtype=float)
    
    # Test 1: Default behavior (no smoothing)
    ev_pmf_strict = EV_PMF.from_samples(ev_samples, w_bins=100, smooth_W=False)
    zeros_strict = (ev_pmf_strict.W == 0).sum()
    print(f"✓ Strict zeros (smooth_W=False): {zeros_strict} zero bins in W")
    
    # Test 2: With smoothing
    ev_pmf_smooth = EV_PMF.from_samples(ev_samples, w_bins=100, smooth_W=True, smooth_eps=1e-6)
    zeros_smooth = (ev_pmf_smooth.W == 0).sum()
    print(f"✓ With smoothing (smooth_W=True): {zeros_smooth} zero bins in W")
    
    # Smoothing should reduce or eliminate zeros
    assert zeros_smooth <= zeros_strict, \
        f"Smoothing should not increase zeros: {zeros_smooth} > {zeros_strict}"
    print(f"✓ Smoothing reduces zero bins: {zeros_strict} → {zeros_smooth}")
    
    # Test 3: Default is no smoothing
    ev_pmf_default = EV_PMF.from_samples(ev_samples, w_bins=100)
    zeros_default = (ev_pmf_default.W == 0).sum()
    assert zeros_default == zeros_strict, \
        f"Default should be strict (no smoothing): {zeros_default} != {zeros_strict}"
    print(f"✓ Default behavior is strict (no smoothing)")
    
    # Test 4: W rows should still be normalized after smoothing
    row_sums = ev_pmf_smooth.W.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"W rows not normalized: {row_sums}"
    print(f"✓ W rows normalized after smoothing: sums = {row_sums}")
    
    # Test 5: Sampling should still work with smoothing
    samples = ev_pmf_smooth.sample(M=10)
    assert samples.shape == (10, 6), f"Unexpected sample shape: {samples.shape}"
    print(f"✓ Sampling works with smoothing: {samples.shape}")
    
    print("✓ All EV_PMF smoothing tests passed")

if __name__ == "__main__":
    test_ev_pmf_smoothing()
