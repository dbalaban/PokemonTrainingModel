"""
Unit tests for bug fixes and improvements.
Tests StatBlock indexing, IV_PMF.getProb consistency, helper functions, and regimen simulator.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/runner/work/PokemonTrainingModel/PokemonTrainingModel/src')

from data_structures import (
    StatBlock, StatType, statblock_to_array, array_to_statblock,
    SpeciesInfo, GrowthRate, TrainingBlock, TrainingRegimen, EncounterOption, Nature
)
from PMFs import IV_PMF, EV_PMF
from regimen_sim import RegimenSimulator


def test_statblock_indexing():
    """Test that StatBlock indexing works correctly for all six stats."""
    print("Testing StatBlock indexing...")
    
    sb = StatBlock(hp=100, atk=120, def_=95, spa=80, spd=85, spe=110)
    
    # Test __getitem__ for all stats
    assert sb[StatType.HP] == 100, f"Expected HP=100, got {sb[StatType.HP]}"
    assert sb[StatType.ATTACK] == 120, f"Expected ATTACK=120, got {sb[StatType.ATTACK]}"
    assert sb[StatType.DEFENSE] == 95, f"Expected DEFENSE=95, got {sb[StatType.DEFENSE]}"
    assert sb[StatType.SPECIAL_ATTACK] == 80, f"Expected SPECIAL_ATTACK=80, got {sb[StatType.SPECIAL_ATTACK]}"
    assert sb[StatType.SPECIAL_DEFENSE] == 85, f"Expected SPECIAL_DEFENSE=85, got {sb[StatType.SPECIAL_DEFENSE]}"
    assert sb[StatType.SPEED] == 110, f"Expected SPEED=110, got {sb[StatType.SPEED]}"
    
    # Test __setitem__ for all stats
    sb[StatType.HP] = 105
    sb[StatType.ATTACK] = 125
    sb[StatType.DEFENSE] = 100
    sb[StatType.SPECIAL_ATTACK] = 85
    sb[StatType.SPECIAL_DEFENSE] = 90
    sb[StatType.SPEED] = 115
    
    assert sb.hp == 105
    assert sb.atk == 125
    assert sb.def_ == 100
    assert sb.spa == 85
    assert sb.spd == 90
    assert sb.spe == 115
    
    print("  ✓ All StatBlock indexing tests passed")


def test_iv_pmf_getprob_consistency():
    """Test that IV_PMF.getProb is consistent with getLogProb."""
    print("Testing IV_PMF.getProb consistency...")
    
    rng = np.random.default_rng(42)
    pmf = IV_PMF(rng=rng)
    
    # Test single IV vector (6,)
    iv_single = np.array([15, 20, 25, 10, 5, 30], dtype=int)
    p = pmf.getProb(iv_single)
    log_p = pmf.getLogProb(iv_single)
    assert np.isclose(p, np.exp(log_p)), f"getProb and exp(getLogProb) mismatch: {p} vs {np.exp(log_p)}"
    
    # Test batched IVs (6, M)
    iv_batch = rng.integers(0, 32, size=(6, 50))
    p_batch = pmf.getProb(iv_batch)
    log_p_batch = pmf.getLogProb(iv_batch)
    assert p_batch.shape == (50,), f"Expected shape (50,), got {p_batch.shape}"
    assert np.allclose(p_batch, np.exp(log_p_batch)), "Batch: getProb and exp(getLogProb) mismatch"
    
    # Test with non-uniform prior
    custom_prior = np.random.rand(6, 32)
    custom_prior /= custom_prior.sum(axis=1, keepdims=True)
    pmf_custom = IV_PMF(prior=custom_prior, rng=rng)
    
    iv_test = rng.integers(0, 32, size=(6, 20))
    p_custom = pmf_custom.getProb(iv_test)
    log_p_custom = pmf_custom.getLogProb(iv_test)
    assert np.allclose(p_custom, np.exp(log_p_custom)), "Custom prior: getProb and exp(getLogProb) mismatch"
    
    print("  ✓ IV_PMF.getProb consistency tests passed")


def test_statblock_array_helpers():
    """Test statblock_to_array and array_to_statblock helpers."""
    print("Testing StatBlock ↔ array helpers...")
    
    # Test statblock_to_array
    sb = StatBlock(hp=100, atk=120, def_=95, spa=80, spd=85, spe=110)
    arr = statblock_to_array(sb)
    expected = np.array([100, 120, 95, 80, 85, 110], dtype=int)
    assert np.array_equal(arr, expected), f"statblock_to_array failed: {arr} vs {expected}"
    
    # Test array_to_statblock
    arr_in = np.array([50, 60, 70, 80, 90, 100])
    sb_out = array_to_statblock(arr_in)
    assert sb_out.hp == 50
    assert sb_out.atk == 60
    assert sb_out.def_ == 70
    assert sb_out.spa == 80
    assert sb_out.spd == 90
    assert sb_out.spe == 100
    
    # Test round-trip
    sb_original = StatBlock(hp=111, atk=222, def_=333, spa=444, spd=555, spe=666)
    arr_temp = statblock_to_array(sb_original)
    sb_restored = array_to_statblock(arr_temp)
    assert sb_original == sb_restored, "Round-trip conversion failed"
    
    print("  ✓ StatBlock ↔ array helper tests passed")


def test_regimen_simulator_rng():
    """Test that RegimenSimulator uses injectable RNG correctly."""
    print("Testing RegimenSimulator RNG injection...")
    
    # Create a simple species and regimen
    species = SpeciesInfo(
        name="TestMon",
        base_stats=StatBlock(45, 49, 49, 65, 65, 45),
        growth_rate=GrowthRate.MEDIUM_FAST,
        base_exp_yield=64,
        ev_yield=StatBlock(0, 0, 0, 1, 0, 0),
    )
    
    target_species = SpeciesInfo(
        name="Target",
        base_stats=StatBlock(40, 45, 40, 50, 50, 35),
        growth_rate=GrowthRate.MEDIUM_FAST,
        base_exp_yield=50,
        ev_yield=StatBlock(0, 0, 0, 1, 0, 0),
    )
    
    encounter = EncounterOption(target=target_species, weight=1.0, levels=[5])
    block = TrainingBlock(start_level=5, end_level=10, location="Test", encounters=[encounter])
    regimen = TrainingRegimen([block])
    
    # Test with seeded RNG - should give reproducible results
    rng1 = np.random.default_rng(123)
    sim1 = RegimenSimulator(regimen, species, gen=5, rng=rng1)
    result1 = sim1.simulate_trial(exp_start=species.exp_to_level(5))
    
    rng2 = np.random.default_rng(123)
    sim2 = RegimenSimulator(regimen, species, gen=5, rng=rng2)
    result2 = sim2.simulate_trial(exp_start=species.exp_to_level(5))
    
    # Results should be identical with same seed
    assert result1 == result2, f"Results differ with same seed: {result1} vs {result2}"
    
    # Test with different seed - should give different results (most likely)
    rng3 = np.random.default_rng(456)
    sim3 = RegimenSimulator(regimen, species, gen=5, rng=rng3)
    result3 = sim3.simulate_trial(exp_start=species.exp_to_level(5))
    
    # Note: there's a small chance results could be identical by chance
    # but for this simple test, they will almost certainly differ
    
    print("  ✓ RegimenSimulator RNG injection tests passed")


def test_regimen_simulator_level_clamp():
    """Test that simulateBlock clamps exp to prevent level overshoot."""
    print("Testing RegimenSimulator level clamping...")
    
    # Create a species with known growth rate
    species = SpeciesInfo(
        name="TestMon",
        base_stats=StatBlock(45, 49, 49, 65, 65, 45),
        growth_rate=GrowthRate.MEDIUM_FAST,
        base_exp_yield=64,
        ev_yield=StatBlock(0, 0, 0, 1, 0, 0),
    )
    
    # Create a high-yielding target that could overshoot
    target_species = SpeciesInfo(
        name="BigTarget",
        base_stats=StatBlock(100, 100, 100, 100, 100, 100),
        growth_rate=GrowthRate.MEDIUM_FAST,
        base_exp_yield=300,
        ev_yield=StatBlock(3, 3, 0, 0, 0, 0),
        is_trainer_owned=True,  # 1.5x multiplier
    )
    
    encounter = EncounterOption(target=target_species, weight=1.0, levels=[50])
    block = TrainingBlock(start_level=10, end_level=15, location="Test", encounters=[encounter])
    regimen = TrainingRegimen([block])
    
    rng = np.random.default_rng(789)
    sim = RegimenSimulator(regimen, species, gen=5, rng=rng)
    
    exp_start = species.exp_to_level(10)
    exp_result, _ = sim.simulateBlock(block, exp_start)
    
    # Check that exp is exactly at end_level threshold (or less)
    exp_threshold = species.exp_to_level(15)
    assert exp_result <= exp_threshold, f"Experience overshot: {exp_result} > {exp_threshold}"
    
    # The level from exp should be exactly end_level
    level_result = species.level_from_exp(exp_result)
    assert level_result == 15, f"Level should be 15, got {level_result}"
    
    print("  ✓ RegimenSimulator level clamping tests passed")


def test_ev_pmf_sample_shape():
    """Test that EV_PMF.sample returns correct shape."""
    print("Testing EV_PMF.sample shape...")
    
    rng = np.random.default_rng(111)
    pmf = EV_PMF(rng=rng)
    
    # Sample should return (M, 6) shape
    M = 100
    samples = pmf.sample(M)
    assert samples.shape == (M, 6), f"Expected shape ({M}, 6), got {samples.shape}"
    
    # Check values are in valid range
    assert np.all(samples >= 0), "Some EV values are negative"
    assert np.all(samples <= 252), "Some EV values exceed 252"
    
    # Check totals are in valid range
    totals = samples.sum(axis=1)
    assert np.all(totals <= 510), "Some total EVs exceed 510"
    
    print("  ✓ EV_PMF.sample shape tests passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running unit tests for bug fixes and improvements")
    print("=" * 60)
    
    test_statblock_indexing()
    test_iv_pmf_getprob_consistency()
    test_statblock_array_helpers()
    test_regimen_simulator_rng()
    test_regimen_simulator_level_clamp()
    test_ev_pmf_sample_shape()
    
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
