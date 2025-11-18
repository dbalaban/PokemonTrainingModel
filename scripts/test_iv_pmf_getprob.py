#!/usr/bin/env python3
"""
Unit test for IV_PMF.getProb fix.
Validates that getProb and getLogProb handle both scalar and batched IV arrays.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from PMFs import IV_PMF

def test_iv_pmf_getprob():
    """Test that IV_PMF.getProb works for both scalar and batched inputs."""
    rng = np.random.default_rng(42)
    iv_pmf = IV_PMF(rng=rng)
    
    # Test scalar input (6,)
    iv_single = np.array([15, 20, 10, 31, 0, 25], dtype=int)
    prob_single = iv_pmf.getProb(iv_single)
    logprob_single = iv_pmf.getLogProb(iv_single)
    
    assert isinstance(prob_single, (float, np.floating)), f"Expected scalar, got {type(prob_single)}"
    assert isinstance(logprob_single, (float, np.floating)), f"Expected scalar, got {type(logprob_single)}"
    assert np.isclose(prob_single, np.exp(logprob_single)), f"getProb != exp(getLogProb) for scalar"
    print(f"✓ Scalar input: P={prob_single:.6f}, log(P)={logprob_single:.6f}")
    
    # Test batched input (6, M)
    M = 5
    iv_batch = rng.integers(0, 32, size=(6, M))
    prob_batch = iv_pmf.getProb(iv_batch)
    logprob_batch = iv_pmf.getLogProb(iv_batch)
    
    assert prob_batch.shape == (M,), f"Expected shape ({M},), got {prob_batch.shape}"
    assert logprob_batch.shape == (M,), f"Expected shape ({M},), got {logprob_batch.shape}"
    assert np.allclose(prob_batch, np.exp(logprob_batch)), f"getProb != exp(getLogProb) for batch"
    print(f"✓ Batched input (6, {M}): shapes match, probabilities consistent")
    
    # Verify consistency: getProb should match individual computations
    for i in range(M):
        iv_i = iv_batch[:, i]
        prob_i_direct = iv_pmf.getProb(iv_i)
        prob_i_from_batch = prob_batch[i]
        assert np.isclose(prob_i_direct, prob_i_from_batch), \
            f"Batch prob inconsistent at index {i}: {prob_i_direct} != {prob_i_from_batch}"
    print(f"✓ Batch probabilities match individual computations")
    
    print("✓ All IV_PMF.getProb tests passed")

if __name__ == "__main__":
    test_iv_pmf_getprob()
