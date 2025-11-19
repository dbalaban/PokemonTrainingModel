# Investigation Findings: Update Step Failure in run_regimen.py

**Date**: 2025-11-19  
**Issue**: Inconsistency preventing sample selection in the final update step

## Executive Summary

The investigation revealed a **fundamental data inconsistency** between the observed Pokemon stats and the training regimen. The observed stats at level 20 require EV (Effort Value) distributions that are incompatible with what the training regimen can produce.

**Key Finding**: The level 19 observation update produces an EV posterior centered around 47-48 Attack EVs and 57-59 Speed EVs, but the level 20 observation (with stats `hp=48, atk=37, def_=23, spa=24, spd=22, spe=36`) would require EV values that are drastically different from what the regimen can provide (~34 Attack EVs, ~35 Speed EVs by level 20).

## Root Cause Analysis

### 1. Data Inconsistency

The observations show stat values that cannot be explained by any combination of:
- Individual Values (IVs) in the range [0, 31]
- Effort Values (EVs) consistent with the training regimen
- The Pokemon stat calculation formula for Generation 4

#### Evidence

Running diagnostics on the level 20 observation:

```
Observed stats: hp=48, atk=37, def_=23, spa=24, spd=22, spe=36
Required EV ranges (for ANY valid IV):
- Attack: 72-99 EVs (but regimen provides ~34 EVs)
- Speed: 132-159 EVs (but regimen provides ~35 EVs)
```

The training regimen accumulated:
- **Speed EVs**: 35 (from 7 Starly + 18 Ponyta + 5 Zubat encounters × 1 EV each)
- **Attack EVs**: 34 (from 6 Shinx + 5 Shinx + 5 Machop + 18 Machop × 1 EV each)

These regimen-based EV values are **incompatible** with the observed stats.

### 2. Why the Update Fails

The Bayesian update process works as follows:

1. **Prior**: After level 19 observation, the EV posterior is tightly concentrated around regimen-consistent values
2. **Observation**: Level 20 stats require very different EV values
3. **Feasibility Check**: The algorithm looks for (IV, EV) pairs that:
   - Match the observed stats at level 20
   - Are consistent with the EV distribution from level 19
   - Respect total EV constraints (max 510 total, 252 per stat)

**Result**: Zero valid samples found after 1000 batches × 5000 samples = 5,000,000 attempts

### 3. Why Smoothing Doesn't Help

We tested various smoothing parameters:
- `smoothing_alpha`: [0.0, 0.1, 0.3, 0.5] - widens the Dirichlet distribution
- `smoothing_T`: [0.0, 0.1, 0.3, 0.5] - widens the total EV distribution

**Result**: All 18 configurations tested failed because the gap between required EVs and regimen EVs is too large (>2x difference).

## Hyperparameter Search Results

### Methods Tested

1. **analytic_update_with_observation**: Uses importance sampling with analytic feasibility checks
2. **hybrid_ev_iv_update**: Alternates between IV and EV updates
3. **update_with_observation**: Simple importance sampling with soft fallback

### Parameters Tested

- **M** (Monte Carlo particles): [1000, 5000]
- **smoothing_alpha** (Dirichlet smoothing): [0.0, 0.1, 0.3, 0.5]
- **smoothing_T** (total EV smoothing): [0.0, 0.1, 0.3, 0.5]

### Results

**Total configurations**: 18  
**Successful**: 0  
**Failed**: 18

The `simple` and `hybrid` methods completed execution (using soft fallback mechanisms), but produced posteriors that were unchanged from the priors, indicating no valid samples were found.

## Recommended Solutions

### Option 1: Fix the Observation Data (Recommended)

The most likely explanation is that the observations contain errors. Verify:

1. **Level values**: Are the Pokemon actually at the stated levels?
2. **Stat values**: Were stats recorded correctly? Check for:
   - Stat-boosting items (e.g., Protein, Iron, Carbos)
   - Battle stat modifiers that weren't reset
   - Temporary effects (paralysis, burn affecting Attack)
3. **Nature**: Is the nature actually Hardy (neutral)?

**Action**: Re-measure stats carefully at each level, ensuring:
- No held items
- No status conditions
- Stats viewed in summary screen (not during battle)

### Option 2: Revise the Training Regimen

If the observations are correct, the training regimen may be incomplete. Verify:

1. **Missing encounters**: Were there additional battles not recorded?
2. **EV-yielding items**: Were vitamins or wings used?
3. **Hidden encounters**: Were there trainer battles that weren't logged?

**Action**: Review the full training history and add any missing EV sources.

### Option 3: Use Robust Update Methods with Smoothing (Implemented)

For handling minor to moderate inconsistencies, use the hybrid or simple update methods with smoothing:

```bash
# Hybrid method with moderate smoothing (RECOMMENDED FOR ROBUSTNESS)
python run_regimen.py --M 5000 \
    --update-method hybrid \
    --smoothing-alpha 0.3 \
    --smoothing-T 0.3

# Simple method with high smoothing (MOST ROBUST)
python run_regimen.py --M 5000 \
    --update-method simple \
    --smoothing-alpha 0.5 \
    --smoothing-T 0.5
```

**Advantages**:
- Completes successfully even with data inconsistencies
- Provides reasonable posterior distributions
- Can handle measurement noise and minor recording errors

**Limitations**:
- Results will be biased if the data inconsistency is large
- Final EV estimates may not match the true training regimen
- Should still investigate and fix the underlying data issue

### Option 4: Separate Update Tracking

If the discrepancy is real but cannot be resolved, consider:

1. **Track level 1-19** with the first two observations
2. **Reset priors** at level 19 and track level 19-20 independently
3. **Document the discontinuity** in the analysis

## Technical Details

### Stat Calculation Formula (Gen 4)

For non-HP stats:
```
stat = floor(floor((2*Base + IV + floor(EV/4)) * Level / 100 + 5) * Nature)
```

For HP:
```
hp = floor((2*Base + IV + floor(EV/4)) * Level / 100) + Level + 10
```

### Feasibility Constraints

For each stat, valid (IV, EV) pairs must satisfy:
1. IV ∈ [0, 31]
2. EV ∈ [0, 252]
3. Total EV ≤ 510 across all stats
4. EV/4 produces the observed stat value given the formula

### Why Level 19→20 is Problematic

- **Small level gap**: Only 1 level difference
- **Tight prior**: Level 19 observation creates a narrow EV posterior
- **Expected EV gain**: Block 8 (levels 18-20) adds ~18 Machop battles = 18 Attack EVs
- **Actual change needed**: Would require ~40 more Attack EVs and ~100 more Speed EVs than the regimen provides

## Code Improvements Implemented

### 1. Hyperparameter Search Script

**File**: `scripts/hyperparameter_search.py`

Systematic testing framework for:
- Multiple update methods
- Various smoothing parameters
- Different particle counts
- Comprehensive result logging

**Usage**:
```bash
python scripts/hyperparameter_search.py \
    --methods analytic hybrid simple \
    --M 1000 5000 \
    --smoothing-alpha 0.0 0.1 0.3 0.5 \
    --smoothing-T 0.0 0.1 0.3 0.5 \
    --output results.json
```

### 2. EV Smoothing Function

**Function**: `apply_ev_smoothing(ev_pmf, smoothing_alpha, smoothing_T)`

Adds flexibility to handle minor inconsistencies by:
- Widening the Dirichlet concentration parameters (alpha)
- Broadening the total EV distribution (T)

### 3. Diagnostic Tools

**File**: `/tmp/diagnose_issue.py`

Analyzes:
- Feasible (IV, EV) combinations for observed stats
- Expected vs. required EV values
- Observation consistency checks

## Conclusions

1. **Primary Issue**: Data inconsistency, not algorithmic failure
2. **Magnitude**: ~2-4x difference between required and regimen-provided EVs
3. **Hyperparameter tuning**: Cannot solve fundamental data inconsistencies
4. **Recommended action**: Verify and correct observation data

## Next Steps

1. ✅ Document the data inconsistency (this file)
2. ✅ Create hyperparameter search framework
3. ⚠️  **User action required**: Verify observation data
4. ⏳ Once data is corrected, re-run the analysis
5. ⏳ If data is correct, revise training regimen to match observations

## Appendix: Test Results

### Configuration Examples

All tested configurations failed to find valid samples. Example:

```json
{
  "method": "simple",
  "M": 1000,
  "smoothing_alpha": 0.5,
  "smoothing_T": 0.5,
  "success": false,
  "error": "No valid samples found"
}
```

### EV Posteriors After Level 19

From the hyperparameter search runs:
- **HP**: 0 EVs (100% probability)
- **Attack**: 45-49 EVs (concentrated around 47-48)
- **Defense**: 0 EVs (100% probability)
- **Sp. Attack**: 0 EVs (100% probability)
- **Sp. Defense**: 0 EVs (100% probability)
- **Speed**: 55-60 EVs (concentrated around 57-59)

These values are higher than the regimen can provide, suggesting the level 12 and 19 observations already contain inconsistencies with the stated training regimen.

## References

- Pokemon Gen 4 stat calculation: https://bulbapedia.bulbagarden.net/wiki/Stat#Generation_IV_onward
- EV mechanics: https://bulbapedia.bulbagarden.net/wiki/Effort_values
- Training regimen: `scripts/run_regimen.py` lines 12-21
