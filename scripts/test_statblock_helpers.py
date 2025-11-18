#!/usr/bin/env python3
"""
Unit test for StatBlock ↔ array helper functions.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from data_structures import StatBlock, statblock_to_array, array_to_statblock

def test_statblock_helpers():
    """Test statblock_to_array and array_to_statblock."""
    # Create a StatBlock
    sb = StatBlock(hp=100, atk=120, def_=80, spa=90, spd=70, spe=110)
    
    # Convert to array
    arr = statblock_to_array(sb)
    assert arr.shape == (6,), f"Expected shape (6,), got {arr.shape}"
    assert arr.dtype == np.int_, f"Expected dtype int, got {arr.dtype}"
    assert np.array_equal(arr, [100, 120, 80, 90, 70, 110]), f"Unexpected array values: {arr}"
    print(f"✓ statblock_to_array: {arr}")
    
    # Convert back to StatBlock
    sb2 = array_to_statblock(arr)
    assert sb2.hp == 100
    assert sb2.atk == 120
    assert sb2.def_ == 80
    assert sb2.spa == 90
    assert sb2.spd == 70
    assert sb2.spe == 110
    print(f"✓ array_to_statblock: hp={sb2.hp}, atk={sb2.atk}, def_={sb2.def_}, spa={sb2.spa}, spd={sb2.spd}, spe={sb2.spe}")
    
    # Test round-trip
    arr2 = statblock_to_array(sb2)
    assert np.array_equal(arr, arr2), f"Round-trip failed: {arr} != {arr2}"
    print(f"✓ Round-trip conversion successful")
    
    print("✓ All StatBlock helper tests passed")

if __name__ == "__main__":
    test_statblock_helpers()
