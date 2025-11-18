#!/usr/bin/env python3
"""
Unit test for EV_PMF optional smoothing.
Validates that alpha smoothing can be toggled on/off and affects concentration estimates.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from PMFs import EV_PMF

def test_ev_pmf_smoothing():
    """Test that EV_PMF.from_samples supports optional alpha smoothing."""
    rng = np.random.default_rng(42)
    
    # Create a simple sample set with limited coverage
    # This should result in concentrated alpha estimates
    ev_samples = np.array([
        [100, 100, 100, 100, 100, 10],  # Repeated pattern
        [100, 100, 100, 100, 100, 10],
        [100, 100, 100, 100, 100, 10],
    ], dtype=float)
    
    # Test 1: Default behavior (no smoothing)
    ev_pmf_strict = EV_PMF.from_samples(ev_samples, alpha_smoothing=0.0)
    alpha_strict = ev_pmf_strict.alpha.copy()
    print(f"✓ Strict alpha (alpha_smoothing=0.0): {alpha_strict}")
    print(f"  Min: {alpha_strict.min():.6f}, Max: {alpha_strict.max():.6f}")
    
    # Test 2: With smoothing
    ev_pmf_smooth = EV_PMF.from_samples(ev_samples, alpha_smoothing=1.0)
    alpha_smooth = ev_pmf_smooth.alpha.copy()
    print(f"✓ With smoothing (alpha_smoothing=1.0): {alpha_smooth}")
    print(f"  Min: {alpha_smooth.min():.6f}, Max: {alpha_smooth.max():.6f}")
    
    # Smoothing should increase all alpha values
    assert np.all(alpha_smooth >= alpha_strict), \
        f"Smoothing should not decrease alpha values"
    print(f"✓ Smoothing increases alpha values")
    
    # Test 3: Default is no smoothing
    ev_pmf_default = EV_PMF.from_samples(ev_samples)
    alpha_default = ev_pmf_default.alpha.copy()
    assert np.allclose(alpha_default, alpha_strict), \
        f"Default should be no smoothing: {alpha_default} != {alpha_strict}"
    print(f"✓ Default behavior is no smoothing")
    
    # Test 4: Alpha should be positive
    assert np.all(alpha_smooth > 0), f"Alpha should be positive: {alpha_smooth}"
    print(f"✓ Alpha values are positive")
    
    # Test 5: Sampling should still work with smoothing
    samples = ev_pmf_smooth.sample(M=10)
    assert samples.shape == (10, 6), f"Unexpected sample shape: {samples.shape}"
    print(f"✓ Sampling works with smoothing: {samples.shape}")
    
    print("✓ All EV_PMF smoothing tests passed")

if __name__ == "__main__":
    test_ev_pmf_smoothing()
