# EV_PMF Mode-Agnostic Refactoring

## Overview
This document describes the refactoring done to make EV_PMF usage mode-agnostic, supporting both 'dirichlet' and 'histogram' modes throughout the codebase.

## Problem Statement
The original code assumed `dirichlet` mode for EV_PMF and would crash when `histogram` mode was active, particularly in `run_regimen.py`:

```
TypeError: unsupported operand type(s) for *: 'NoneType' and 'NoneType'
  at bayesian_model.py:33 in update_ev_pmf
  new_T = np.convolve(prior.T, upd.T)[:prior.max_total_ev + 1]
```

The issue occurred because:
- `histogram` mode sets `T` and `alpha` to `None`
- `update_ev_pmf` directly accessed `prior.T` and `upd.T` without checking mode
- Other functions similarly assumed dirichlet-specific attributes

## Solution

### 1. Mode-Agnostic Core Functions

#### update_ev_pmf()
**File**: `src/bayesian_model.py`

Made the function check mode and handle each appropriately:

**Dirichlet mode:**
- Convolves T distributions: `new_T = np.convolve(prior.T, upd.T)`
- Combines alpha parameters using weighted average

**Histogram mode:**
- Convolves independent histograms per stat
- Each stat histogram is convolved separately
- Truncates at max_ev and renormalizes

#### apply_ev_smoothing()
**File**: `src/stat_tracker.py`

**Dirichlet mode:**
- Smooths alpha: mixes with uniform concentration
- Smooths T: mixes with uniform distribution around expected total

**Histogram mode:**
- Applies Gaussian blur to each histogram
- Kernel width based on smoothing parameter

#### EV_PMF.getProb()
**File**: `src/PMFs.py`

**Dirichlet mode:**
- Calculates: `log P(EV) = log P(T) + log Dirichlet(W6 | alpha)`
- Uses scipy.special.gammaln for Dirichlet density

**Histogram mode:**
- Calculates: `log P(EV) = sum_i log P(EV_i)`
- Product of independent marginal probabilities

### 2. Mode Auto-Detection

**File**: `src/PMFs.py` - EV_PMF.__init__

The constructor now auto-detects mode from parameters:

```python
if mode is None:
    if histograms is not None:
        mode = 'histogram'
    elif priorT is not None or alpha is not None:
        mode = 'dirichlet'
    else:
        mode = 'dirichlet'  # Default for backward compatibility
```

This ensures that passing `priorT` and `alpha` automatically uses dirichlet mode.

### 3. Mode Preservation

#### analytic_update_with_observation()
**File**: `src/bayesian_model.py`

Updated to preserve the input mode:
```python
new_ev_pmf = EV_PMF.from_samples(EV_all.T, weights=w, mode=prior_ev.mode, rng=prior_ev.rng)
```

#### RegimenSimulator.toPMF()
**File**: `src/regimen_sim.py`

Added mode parameter:
```python
def toPMF(self, allocator="multinomial", mode="dirichlet") -> EV_PMF:
    return EV_PMF.from_samples(ev_array, allocator=allocator, mode=mode)
```

#### track_training_stats()
**File**: `src/stat_tracker.py`

Passes mode through the pipeline:
```python
post_ev_sim = simulator.toPMF(allocator="round", mode=current_ev_pmf.mode)
```

### 4. Mode Compatibility Checks

Added validation for methods that only support dirichlet mode:

**hybrid_ev_iv_update()** and **update_with_observation()**:
```python
if prior_ev.mode != 'dirichlet':
    raise ValueError(
        f"This method only supports EV_PMF in 'dirichlet' mode. "
        f"Current mode: '{prior_ev.mode}'. "
        f"Consider using 'analytic' update method instead."
    )
```

### 5. User-Facing Interface

#### track_training_stats()
**File**: `src/stat_tracker.py`

Added `ev_mode` parameter:
```python
def track_training_stats(
    ...
    ev_mode: str = 'dirichlet',
) -> Tuple[EV_PMF, IV_PMF]:
```

Validates mode/method compatibility:
```python
if ev_mode == 'histogram' and update_method in ['hybrid', 'simple']:
    raise ValueError(
        f"Update method '{update_method}' only supports 'dirichlet' mode. "
        f"For 'histogram' mode, use 'analytic' update method."
    )
```

#### run_regimen.py
**File**: `scripts/run_regimen.py`

Added command-line argument:
```python
parser.add_argument('--ev-mode', type=str, default='dirichlet',
                    choices=['dirichlet', 'histogram'],
                    help='EV PMF mode (default: dirichlet). Note: histogram mode only works with analytic update method.')
```

## Mode Support Summary

| Function/Method | Dirichlet | Histogram | Notes |
|----------------|-----------|-----------|-------|
| `update_ev_pmf` | ✅ | ✅ | Fully mode-agnostic |
| `apply_ev_smoothing` | ✅ | ✅ | Different smoothing per mode |
| `EV_PMF.getProb` | ✅ | ✅ | Different probability calculations |
| `analytic_update_with_observation` | ✅ | ✅ | Preserves input mode |
| `hybrid_ev_iv_update` | ✅ | ❌ | Raises error for histogram |
| `update_with_observation` | ✅ | ❌ | Raises error for histogram |

## Usage Examples

### Dirichlet Mode (Default)
```bash
python scripts/run_regimen.py --M 1000 --update-method analytic
```

### Histogram Mode
```bash
python scripts/run_regimen.py --M 1000 --update-method analytic --ev-mode histogram
```

### Error: Incompatible Mode/Method
```bash
python scripts/run_regimen.py --M 1000 --update-method hybrid --ev-mode histogram
# ValueError: Update method 'hybrid' only supports 'dirichlet' mode.
```

## Testing Results

All tests passed:
- ✅ `run_regimen.py --ev-mode dirichlet` (default behavior)
- ✅ `run_regimen.py --ev-mode histogram` (new feature)
- ✅ `update_ev_pmf` with both modes
- ✅ Mode auto-detection
- ✅ CodeQL security scan (0 alerts)

## Backward Compatibility

All changes are backward compatible:
- Default mode remains `dirichlet`
- Existing code without explicit mode specification continues to work
- Mode auto-detection ensures proper mode selection based on constructor arguments

## Future Work

1. **Extend histogram support to hybrid/simple updates**: Currently these methods only support dirichlet mode
2. **Add comprehensive unit tests**: Create test suite covering both modes
3. **Performance optimization**: Profile and optimize histogram convolution for large EVs
4. **Documentation**: Add mode-specific behavior to docstrings

## References

- Issue: "Refactor EV_PMF usage to support mode-agnostic logic (dirichlet/histogram compatibility)"
- Files Modified:
  - `src/bayesian_model.py`
  - `src/stat_tracker.py`
  - `src/PMFs.py`
  - `src/regimen_sim.py`
  - `scripts/run_regimen.py`
