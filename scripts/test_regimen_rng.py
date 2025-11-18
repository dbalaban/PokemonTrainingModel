#!/usr/bin/env python3
"""
Unit test for RegimenSimulator injectable RNG.
Validates that RNG can be injected and produces reproducible results.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from regimen_sim import RegimenSimulator
from data_structures import (
    TrainingRegimen, TrainingBlock, SpeciesInfo, StatBlock,
    GrowthRate, EncounterOption, Nature
)

def test_regimen_simulator_rng():
    """Test that RegimenSimulator accepts injectable RNG for reproducibility."""
    
    # Create a simple species and regimen for testing
    species = SpeciesInfo(
        name="TestMon",
        base_stats=StatBlock(hp=80, atk=95, def_=85, spa=65, spd=65, spe=90),
        growth_rate=GrowthRate.MEDIUM_FAST,
        base_exp_yield=100,
        ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
        is_trainer_owned=False
    )
    
    target_species = SpeciesInfo(
        name="WildMon",
        base_stats=StatBlock(hp=60, atk=70, def_=60, spa=50, spd=50, spe=80),
        growth_rate=GrowthRate.MEDIUM_FAST,
        base_exp_yield=80,
        ev_yield=StatBlock(hp=0, atk=1, def_=0, spa=0, spd=0, spe=0),
        is_trainer_owned=False
    )
    
    encounter = EncounterOption(target=target_species, weight=1.0, levels=[10])
    block = TrainingBlock(start_level=5, end_level=10, location="TestRoute", encounters=[encounter])
    regimen = TrainingRegimen([block])
    
    # Test 1: Default RNG (should use np.random.default_rng())
    sim1 = RegimenSimulator(regimen, species, gen=5)
    assert hasattr(sim1, 'rng'), "RegimenSimulator should have rng attribute"
    assert isinstance(sim1.rng, np.random.Generator), "rng should be a Generator"
    print(f"✓ Default RNG created: {type(sim1.rng)}")
    
    # Test 2: Inject custom RNG with seed for reproducibility
    rng1 = np.random.default_rng(42)
    sim2 = RegimenSimulator(regimen, species, gen=5, rng=rng1)
    assert sim2.rng is rng1, "Injected RNG not used"
    print(f"✓ Custom RNG injected successfully")
    
    # Test 3: Same seed should produce same results
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    
    sim_a = RegimenSimulator(regimen, species, gen=5, rng=rng_a)
    sim_b = RegimenSimulator(regimen, species, gen=5, rng=rng_b)
    
    # Run a simple simulation
    exp_start = species.exp_to_level(5)
    ev_a = sim_a.simulate_trial(exp_start)
    ev_b = sim_b.simulate_trial(exp_start)
    
    # Results should be identical
    assert ev_a.hp == ev_b.hp, f"HP mismatch: {ev_a.hp} != {ev_b.hp}"
    assert ev_a.atk == ev_b.atk, f"ATK mismatch: {ev_a.atk} != {ev_b.atk}"
    print(f"✓ Same seed produces identical results: EV gains = {ev_a.atk} ATK")
    
    # Test 4: Different seeds should (likely) produce different results
    rng_c = np.random.default_rng(456)
    sim_c = RegimenSimulator(regimen, species, gen=5, rng=rng_c)
    ev_c = sim_c.simulate_trial(exp_start)
    
    # Very likely to be different (though not guaranteed for simple cases)
    print(f"✓ Different seed produces result: EV gains = {ev_c.atk} ATK")
    
    print("✓ All RegimenSimulator RNG tests passed")

if __name__ == "__main__":
    test_regimen_simulator_rng()
