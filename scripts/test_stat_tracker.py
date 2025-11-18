#!/usr/bin/env python3
"""
Test for stat_tracker.py module
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from data_structures import (
    TrainingRegimen,
    TrainingBlock,
    EncounterOption,
    ObservedStats,
    StatBlock,
    Nature,
    SpeciesInfo,
    GrowthRate,
)
from stat_tracker import split_regimen_at_levels, track_training_stats


def test_split_regimen_at_levels():
    """Test that split_regimen_at_levels correctly splits blocks at observation levels."""
    print("\n" + "="*60)
    print("Testing split_regimen_at_levels")
    print("="*60)
    
    # Create a simple species for testing
    test_species = SpeciesInfo(
        name="TestMon",
        base_stats=StatBlock(hp=50, atk=50, def_=50, spa=50, spd=50, spe=50),
        ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
        base_exp_yield=50,
        growth_rate=GrowthRate.MEDIUM_FAST,
    )
    
    # Create a regimen with blocks 1-5, 5-10, 10-15
    blocks = [
        TrainingBlock(
            start_level=1,
            end_level=5,
            location="Area 1",
            encounters=[
                EncounterOption(target=test_species, weight=1.0, levels=[2, 3])
            ]
        ),
        TrainingBlock(
            start_level=5,
            end_level=10,
            location="Area 2",
            encounters=[
                EncounterOption(target=test_species, weight=1.0, levels=[4, 5])
            ]
        ),
        TrainingBlock(
            start_level=10,
            end_level=15,
            location="Area 3",
            encounters=[
                EncounterOption(target=test_species, weight=1.0, levels=[6, 7])
            ]
        ),
    ]
    regimen = TrainingRegimen(blocks=blocks)
    
    # Test 1: Split at level 7 (inside second block)
    print("\nTest 1: Split at level 7 (inside block 5-10)")
    split_regimen = split_regimen_at_levels(regimen, [7])
    assert len(split_regimen.blocks) == 4, f"Expected 4 blocks, got {len(split_regimen.blocks)}"
    assert split_regimen.blocks[0].start_level == 1 and split_regimen.blocks[0].end_level == 5
    assert split_regimen.blocks[1].start_level == 5 and split_regimen.blocks[1].end_level == 7
    assert split_regimen.blocks[2].start_level == 7 and split_regimen.blocks[2].end_level == 10
    assert split_regimen.blocks[3].start_level == 10 and split_regimen.blocks[3].end_level == 15
    print("  ✓ Block split correctly at level 7")
    
    # Test 2: Split at multiple levels
    print("\nTest 2: Split at levels 3, 7, 12")
    split_regimen = split_regimen_at_levels(regimen, [3, 7, 12])
    assert len(split_regimen.blocks) == 6
    print(f"  ✓ Created {len(split_regimen.blocks)} blocks from 3 original blocks")
    
    # Verify all blocks are contiguous
    for i in range(len(split_regimen.blocks) - 1):
        assert split_regimen.blocks[i].end_level == split_regimen.blocks[i + 1].start_level
    print("  ✓ All blocks are contiguous")
    
    # Test 3: Split at boundary levels (should not create new blocks)
    print("\nTest 3: Split at boundary levels 5, 10")
    split_regimen = split_regimen_at_levels(regimen, [5, 10])
    assert len(split_regimen.blocks) == 3, f"Expected 3 blocks (no splits), got {len(split_regimen.blocks)}"
    print("  ✓ No unnecessary splits at boundaries")
    
    # Test 4: Empty split list
    print("\nTest 4: Empty split list")
    split_regimen = split_regimen_at_levels(regimen, [])
    assert len(split_regimen.blocks) == 3
    print("  ✓ Returns original regimen when no splits requested")
    
    print("\n" + "="*60)
    print("split_regimen_at_levels tests PASSED ✓")
    print("="*60)


def test_track_training_stats_basic():
    """Test that track_training_stats runs without errors on a simple example."""
    print("\n" + "="*60)
    print("Testing track_training_stats basic functionality")
    print("="*60)
    
    # Create a simple species
    test_species = SpeciesInfo(
        name="TestMon",
        base_stats=StatBlock(hp=50, atk=50, def_=50, spa=50, spd=50, spe=50),
        ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
        base_exp_yield=50,
        growth_rate=GrowthRate.MEDIUM_FAST,
    )
    
    # Create a simple regimen
    regimen = TrainingRegimen(blocks=[
        TrainingBlock(
            start_level=5,
            end_level=10,
            location="Test Area",
            encounters=[
                EncounterOption(target=test_species, weight=1.0, levels=[5, 6])
            ]
        ),
    ])
    
    # Create observation at level 10
    observations = [
        ObservedStats(
            level=10,
            stats=StatBlock(hp=30, atk=25, def_=22, spa=21, spd=20, spe=28)
        ),
    ]
    
    # Neutral nature
    nature = Nature(name="Hardy", inc=None, dec=None)
    
    print("\nRunning track_training_stats with M=500...")
    
    # Run the tracker
    final_ev_pmf, final_iv_pmf = track_training_stats(
        regimen=regimen,
        observations=observations,
        base_stats=test_species.base_stats,
        nature=nature,
        species_info=test_species,
        gen=4,
        M=500,
        verbose=False,
        debug_plots=False,
    )
    
    print("  ✓ track_training_stats completed without errors")
    
    # Check that PMFs are valid
    assert final_iv_pmf.P.shape == (6, 32), "IV PMF has wrong shape"
    assert np.allclose(final_iv_pmf.P.sum(axis=1), 1.0), "IV PMF not normalized"
    print("  ✓ IV PMF is properly normalized")
    
    assert final_ev_pmf.T.shape[0] == 511, "EV total PMF has wrong shape"
    assert np.isclose(final_ev_pmf.T.sum(), 1.0), "EV total PMF not normalized"
    print("  ✓ EV PMF is properly normalized")
    
    print("\n" + "="*60)
    print("track_training_stats basic tests PASSED ✓")
    print("="*60)


def test_validation_errors():
    """Test that track_training_stats properly validates inputs."""
    print("\n" + "="*60)
    print("Testing track_training_stats input validation")
    print("="*60)
    
    # Create a simple species
    test_species = SpeciesInfo(
        name="TestMon",
        base_stats=StatBlock(hp=50, atk=50, def_=50, spa=50, spd=50, spe=50),
        ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
        base_exp_yield=50,
        growth_rate=GrowthRate.MEDIUM_FAST,
    )
    
    regimen = TrainingRegimen(blocks=[
        TrainingBlock(
            start_level=5,
            end_level=10,
            location="Test Area",
            encounters=[
                EncounterOption(target=test_species, weight=1.0, levels=[5, 6])
            ]
        ),
    ])
    
    nature = Nature(name="Hardy", inc=None, dec=None)
    
    # Test: observation level outside regimen range
    print("\nTest 1: Observation level outside regimen range")
    observations = [
        ObservedStats(
            level=15,  # Outside range [5, 10]
            stats=StatBlock(hp=30, atk=25, def_=22, spa=21, spd=20, spe=28)
        ),
    ]
    
    try:
        track_training_stats(
            regimen=regimen,
            observations=observations,
            base_stats=test_species.base_stats,
            nature=nature,
            species_info=test_species,
            gen=4,
            M=100,
            verbose=False,
        )
        assert False, "Should have raised ValueError for out-of-range observation"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:60]}...")
    
    print("\n" + "="*60)
    print("Validation tests PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    test_split_regimen_at_levels()
    test_track_training_stats_basic()
    test_validation_errors()
    
    print("\n" + "="*60)
    print("ALL stat_tracker TESTS PASSED ✓")
    print("="*60)
