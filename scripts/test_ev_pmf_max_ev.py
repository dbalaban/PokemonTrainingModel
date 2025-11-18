#!/usr/bin/env python3
"""
Unit test for EV_PMF.MAX_EV class constant.
Validates that max_ev is a class-level constant shared by all instances.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from PMFs import EV_PMF

def test_ev_pmf_max_ev():
    """Test that EV_PMF.MAX_EV is a class constant and max_ev is a read-only property."""
    
    # Test class constant exists and has correct value
    assert hasattr(EV_PMF, 'MAX_EV'), "EV_PMF.MAX_EV class constant not found"
    assert EV_PMF.MAX_EV == 252, f"Expected MAX_EV=252, got {EV_PMF.MAX_EV}"
    print(f"✓ EV_PMF.MAX_EV = {EV_PMF.MAX_EV}")
    
    # Create two instances
    ev_pmf1 = EV_PMF()
    ev_pmf2 = EV_PMF()
    
    # Test that both instances share the same MAX_EV
    assert ev_pmf1.MAX_EV == ev_pmf2.MAX_EV == 252, "Instances do not share MAX_EV"
    print(f"✓ All instances share MAX_EV = {ev_pmf1.MAX_EV}")
    
    # Test that max_ev property returns MAX_EV
    assert ev_pmf1.max_ev == EV_PMF.MAX_EV, f"max_ev property returns {ev_pmf1.max_ev}, expected {EV_PMF.MAX_EV}"
    assert ev_pmf2.max_ev == EV_PMF.MAX_EV, f"max_ev property returns {ev_pmf2.max_ev}, expected {EV_PMF.MAX_EV}"
    print(f"✓ max_ev property correctly returns MAX_EV = {ev_pmf1.max_ev}")
    
    # Test that max_ev is read-only (trying to set it should fail)
    try:
        ev_pmf1.max_ev = 100
        assert False, "max_ev should be read-only but was set successfully"
    except AttributeError:
        print(f"✓ max_ev is read-only (cannot be set)")
    
    # Test that max_total_ev is correctly computed from MAX_EV
    expected_max_total = 2 * EV_PMF.MAX_EV + 6  # 510
    assert ev_pmf1.max_total_ev == expected_max_total, \
        f"max_total_ev should be {expected_max_total}, got {ev_pmf1.max_total_ev}"
    print(f"✓ max_total_ev correctly computed as {ev_pmf1.max_total_ev}")
    
    print("✓ All EV_PMF.MAX_EV tests passed")

if __name__ == "__main__":
    test_ev_pmf_max_ev()
