#!/usr/bin/env python3
"""
Unit test for StatBlock indexing fix.
Validates that sb[stat] and sb[stat] = x work correctly for all six stats.
"""

import sys
sys.path.insert(0, '../src')

from data_structures import StatBlock, StatType

def test_statblock_indexing():
    """Test that StatBlock can be indexed with StatType enum values."""
    sb = StatBlock(hp=100, atk=120, def_=80, spa=90, spd=70, spe=110)
    
    # Test __getitem__ for all six stats
    assert sb[StatType.HP] == 100, f"Expected HP=100, got {sb[StatType.HP]}"
    assert sb[StatType.ATTACK] == 120, f"Expected ATTACK=120, got {sb[StatType.ATTACK]}"
    assert sb[StatType.DEFENSE] == 80, f"Expected DEFENSE=80, got {sb[StatType.DEFENSE]}"
    assert sb[StatType.SPECIAL_ATTACK] == 90, f"Expected SPECIAL_ATTACK=90, got {sb[StatType.SPECIAL_ATTACK]}"
    assert sb[StatType.SPECIAL_DEFENSE] == 70, f"Expected SPECIAL_DEFENSE=70, got {sb[StatType.SPECIAL_DEFENSE]}"
    assert sb[StatType.SPEED] == 110, f"Expected SPEED=110, got {sb[StatType.SPEED]}"
    
    # Test __setitem__ for all six stats
    sb[StatType.HP] = 105
    sb[StatType.ATTACK] = 125
    sb[StatType.DEFENSE] = 85
    sb[StatType.SPECIAL_ATTACK] = 95
    sb[StatType.SPECIAL_DEFENSE] = 75
    sb[StatType.SPEED] = 115
    
    assert sb.hp == 105, f"Expected HP=105, got {sb.hp}"
    assert sb.atk == 125, f"Expected atk=125, got {sb.atk}"
    assert sb.def_ == 85, f"Expected def_=85, got {sb.def_}"
    assert sb.spa == 95, f"Expected spa=95, got {sb.spa}"
    assert sb.spd == 75, f"Expected spd=75, got {sb.spd}"
    assert sb.spe == 115, f"Expected spe=115, got {sb.spe}"
    
    print("✓ All StatBlock indexing tests passed")

if __name__ == "__main__":
    test_statblock_indexing()
