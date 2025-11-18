# PMFs.py

from __future__ import annotations
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any, List

from data_structures import *  # StatBlock, etc.

# ========= Utilities =========

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / (np.arange(v.size) + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def project_capped_simplex(y: np.ndarray, T: int, U: int = 252, tol: float = 1e-9) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    Uvec = np.full_like(y, float(U))
    T = float(np.clip(T, 0.0, Uvec.sum()))
    lo = (y - Uvec).min() - 1.0
    hi = y.max() + 1.0
    x = None
    for _ in range(60):
        tau = 0.5 * (lo + hi)
        x = np.clip(y - tau, 0.0, U)
        s = float(x.sum())
        if abs(s - T) <= tol: break
        lo, hi = (tau, hi) if s > T else (lo, tau)
    return x if x is not None else np.clip(y, 0.0, U)

def integerize_with_waterfill(x: np.ndarray, T: int, U: int = 252) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    E = np.rint(x).astype(int)
    E = np.clip(E, 0, U)
    diff = int(T) - int(E.sum())
    if diff == 0: return E
    frac = x - np.floor(x)
    order = np.argsort(frac)[::-1] if diff > 0 else np.argsort(frac)
    step = 1 if diff > 0 else -1
    k = 0
    while diff != 0 and k < E.size:
        i = order[k]; cand = E[i] + step
        if 0 <= cand <= U:
            E[i] = cand; diff -= step
        k += 1
    return E

# ========= IV PMF =========

class IV_PMF:
    def __init__(self, prior: np.ndarray | None = None, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        if prior is None:
            self.prior = np.full((6, 32), 1/32.0)
        else:
            arr = np.asarray(prior, dtype=float); assert arr.shape == (6, 32)
            self.prior = arr / arr.sum(axis=1, keepdims=True)

    @property
    def P(self) -> np.ndarray: return self.prior

    def normalize_(self) -> None:
        rs = self.prior.sum(axis=1, keepdims=True)
        np.divide(self.prior, rs, out=self.prior, where=(rs > 0))

    def sample(self, M: int) -> np.ndarray:
        cdf = np.cumsum(self.prior, axis=1)
        cdf[:, -1] = 1.0
        u = self.rng.random((6, M))
        idx = np.empty((6, M), dtype=int)
        for s in range(6):
            idx[s] = np.searchsorted(cdf[s], u[s], side="right")
        return np.clip(idx, 0, 31)

    def getProb(self, IV: np.ndarray) -> np.ndarray:
        """
        Return P(IV) under this PMF.
        
        Parameters
        ----------
        IV : shape (6,) or (6, M) array of integer IVs in [0..31]
             For a single IV vector, use shape (6,).
             For M samples, use shape (6, M).
        
        Returns
        -------
        scalar (if IV is (6,)) or (M,) ndarray of probabilities
        """
        logP = self.getLogProb(IV)
        return np.exp(logP)

    def getLogProb(self, IV: NDArray[np.integer] | NDArray[np.floating]) -> np.ndarray:
        """
        Return log P(IV) under the row-wise prior.
        
        Parameters
        ----------
        IV : shape (6,) or (6, M) array of integer IVs in [0..31]
             For a single IV vector, use shape (6,).
             For M samples, use shape (6, M).
        
        Returns
        -------
        scalar (if IV is (6,)) or (M,) ndarray of log-probabilities
        
        Notes
        -----
        - Zeros in the prior produce -inf (no smoothing).
        - Out-of-range IV indices are treated as probability 0 → -inf.
        """
        idx = np.asarray(IV, dtype=int)
        scalar_input = idx.ndim == 1
        if scalar_input:
            idx = idx[:, None]  # (6,1) for uniform handling

        # log of prior with proper handling of zeros: log(0) -> -inf
        with np.errstate(divide='ignore', invalid='ignore'):
            logP = np.log(self.prior)  # (6, 32)

        # mark out-of-range indices as invalid → -inf
        invalid = (idx < 0) | (idx > 31)
        idx_safe = np.clip(idx, 0, 31)

        # gather per-row log-probs: for each row s and column m, get logP[s, idx_safe[s, m]]
        gathered = np.empty_like(idx, dtype=float)
        for s in range(6):
            gathered[s] = logP[s, idx_safe[s]]

        # force invalid selections to -inf
        gathered = np.where(invalid, -np.inf, gathered)

        # sum across the 6 stats → (M,)
        out = gathered.sum(axis=0)

        # if single vector input, return scalar
        return out[0] if scalar_input else out

    def weighted_add_(self, iv_mat: np.ndarray, w: np.ndarray) -> None:
        out = np.zeros_like(self.prior)
        for s in range(6): np.add.at(out[s], iv_mat[s], w)
        self.prior = out; self.normalize_()

    def blend(self, other: "IV_PMF", mode="linear", alpha: float = 0.5) -> "IV_PMF":
        if mode == "linear":
            P = alpha*self.P + (1-alpha)*other.P
        elif mode == "geometric":
            eps = 1e-12; P = np.exp(alpha*np.log(self.P+eps)+(1-alpha)*np.log(other.P+eps))
        else: raise ValueError("mode must be linear|geometric")
        out = IV_PMF(P, rng=self.rng); out.normalize_(); return out

    @staticmethod
    def from_samples(
        samples: "np.ndarray",
        *,
        weights: "np.ndarray | list[float] | None" = None,
        rng: np.random.Generator | None = None,
        return_hist: bool = False,
    ) -> "IV_PMF | tuple[IV_PMF, dict[str, np.ndarray]]":
        """
        Build an IV_PMF from IV samples, optionally weighted.

        Parameters
        ----------
        samples : (N,6) integer IVs in [0..31]
        weights : optional (N,) nonnegative weights; if None, uniform
        rng      : optional RNG to attach to the IV_PMF
        return_hist : if True, also return {'counts': (6,32), 'weights': (N,)}

        Returns
        -------
        IV_PMF or (IV_PMF, hist_dict)
        """
        IV = np.asarray(samples, dtype=int)
        if IV.ndim != 2 or IV.shape[1] != 6:
            raise ValueError("samples must be (N,6) integer IVs in [0..31].")

        N = IV.shape[0]

        # normalize weights
        if weights is None:
            w = np.full(N, 1.0 / max(N, 1), dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.ndim != 1 or w.shape[0] != N:
                raise ValueError("weights must be a vector of length N.")
            w = np.clip(w, 0.0, np.inf)
            s = w.sum()
            w = (w / s) if s > 0 else np.full(N, 1.0 / max(N, 1), dtype=float)

        # vectorized weighted histograms per stat
        counts = np.zeros((6, 32), dtype=float)
        for s_idx in range(6):
            counts[s_idx] = np.bincount(IV[:, s_idx], weights=w, minlength=32)

        # normalize rows to make a proper PMF
        row_sums = counts.sum(axis=1, keepdims=True)
        P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=(row_sums > 0))

        pmf = IV_PMF(prior=P, rng=rng)

        if not return_hist:
            return pmf
        return pmf, {"counts": counts, "weights": w}

# ========= EV PMF (stick-breaking + mandatory caps) =========

class EV_PMF:
    """
    EV distribution parameterized by:
      - T: PMF over total EVs (0..510)
      - W: 5 independent PMFs over stick-breaking variables s1..s5 in [0,1]

    Allocator:
      - 'multinomial' : draws EV via capped multinomial with iterative repair (sum=T, each ≤ 252)
      - 'round'       : deterministic proportional rounding with caps (vectorized)
    """

    # Class-level constant for per-stat EV cap (shared by all instances)
    MAX_EV: int = 252

    def __init__(
        self,
        priorT: np.ndarray | None = None,
        priorW: np.ndarray | None = None,
        w_bins: int = 506,
        rng: np.random.Generator | None = None,
        *,
        allocator: str = "round",   # 'multinomial' | 'round'
    ):
        self.n_stats = 6
        self.max_total_ev = 2 * self.MAX_EV + self.n_stats  # 510
        self.w_bins = w_bins
        self.rng = rng if rng is not None else np.random.default_rng()
        self.allocator = allocator

        if priorT is not None:
            assert priorT.shape == (self.max_total_ev + 1,); self.T = priorT.astype(float)
        else:
            self.T = np.full(self.max_total_ev + 1, 1.0/(self.max_total_ev + 1), dtype=float)
        self.T /= self.T.sum()

        if priorW is not None:
            assert priorW.shape == (5, self.w_bins); self.W = priorW.astype(float)
        else:
            self.W = np.full((5, self.w_bins), 1.0/self.w_bins, dtype=float)
        row_sums = self.W.sum(axis=1, keepdims=True)
        self.W = np.divide(self.W, row_sums, out=self.W, where=(row_sums > 0))

        self.s_grid = np.linspace(0.0, 1.0, self.w_bins)

    @property
    def max_ev(self) -> int:
        """Read-only property that returns the class-level per-stat EV cap."""
        return self.MAX_EV

    # -- stick-breaking --

    @staticmethod
    def _stick_breaking_from_unit(S5: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        S = np.clip(S5, eps, 1.0 - eps)
        one_minus = 1.0 - S
        cumprod = np.cumprod(one_minus, axis=0)
        W6 = np.empty((6, S.shape[1]), dtype=S.dtype)
        W6[0] = S[0]
        W6[1] = cumprod[0] * S[1]
        W6[2] = cumprod[1] * S[2]
        W6[3] = cumprod[2] * S[3]
        W6[4] = cumprod[3] * S[4]
        W6[5] = cumprod[4]
        return W6

    @staticmethod
    def _invert_stick_breaking(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        S = np.empty(5, dtype=float); rem = 1.0
        for k in range(5):
            s_k = 0.0 if rem <= eps else w[k] / rem
            S[k] = np.clip(s_k, 0.0, 1.0); rem *= (1.0 - S[k])
        return S

    # -- sampling primitives --

    def _sample_S5(self, M: int) -> np.ndarray:
        B = self.w_bins
        row_sums = self.W.sum(axis=1, keepdims=True)
        W_row = np.divide(self.W, row_sums, out=np.zeros_like(self.W), where=(row_sums > 0))
        cdfs = np.cumsum(W_row, axis=1); cdfs[:, -1] = 1.0
        u = self.rng.random((5, M))
        idx = np.empty((5, M), dtype=int)
        for r in range(5): idx[r] = np.searchsorted(cdfs[r], u[r], side='right')
        idx = np.minimum(idx, B - 1)
        return self.s_grid[idx]

    def _sample_totals(self, M: int) -> np.ndarray:
        cdf = np.cumsum(self.T); cdf[-1] = 1.0
        u = self.rng.random(M)
        return np.searchsorted(cdf, u, side='right').clip(0, self.max_total_ev)

    # -- rounding allocator (vectorized) --

    @staticmethod
    def _round_allocations_to_totals(alloc: np.ndarray, totals: np.ndarray, max_ev: int) -> np.ndarray:
        """
        Round fractional EV allocations to integers while enforcing:
        1. Total EV sum equals `totals` for each sample
        2. Per-stat EV does not exceed `max_ev` (typically 252)
        
        Per-stat EV cap enforcement: Any EV value that would exceed max_ev for a stat
        is treated as infeasible and capped at max_ev. This ensures adherence to game
        mechanics where individual stat EVs cannot exceed the per-stat cap.
        
        Note: If the total cannot be achieved due to per-stat caps, the function
        distributes as much as possible without violating caps.
        """
        alloc = np.maximum(alloc, 0.0)
        ev_floor = np.floor(alloc).astype(int)
        # Enforce per-stat cap: values above max_ev are infeasible
        ev_floor = np.clip(ev_floor, 0, max_ev)
        remaining = totals - ev_floor.sum(axis=0)
        if not np.any(remaining): return ev_floor
        frac  = alloc - np.floor(alloc)
        order = np.argsort(-frac, axis=0)
        out = ev_floor.copy()
        for k in range(6):
            if not np.any(remaining > 0): break
            rows = order[k, np.arange(alloc.shape[1])]
            cols = np.arange(alloc.shape[1])
            # Enforce per-stat cap: cannot exceed max_ev
            cap  = max_ev - out[rows, cols]
            inc  = (remaining > 0).astype(int)
            inc  = np.minimum(inc, cap)
            out[rows, cols] += inc
            remaining -= inc
        if np.any(remaining > 0):
            cols = np.where(remaining > 0)[0]
            for r in range(6):
                if cols.size == 0: break
                # Enforce per-stat cap: cannot exceed max_ev
                cap = max_ev - out[r, cols]
                add = np.minimum(cap, remaining[cols])
                out[r, cols] += add
                remaining[cols] -= add
                cols = np.where(remaining > 0)[0]
        return out

    # -- multinomial allocator (mandatory caps via repair) --

    @staticmethod
    def _alloc_multinomial_capped(T_idx: np.ndarray, W6: np.ndarray, U: int, rng: np.random.Generator) -> np.ndarray:
        M = T_idx.shape[0]
        EV = np.zeros((6, M), dtype=int)
        for j in range(M):
            t_rem = int(T_idx[j])
            if t_rem <= 0: continue
            p = W6[:, j].astype(float).copy()
            s = p.sum()
            p = p / s if s > 0 else np.full(6, 1/6.0)
            x = rng.multinomial(t_rem, p, size=1)[0]
            while True:
                over = x - U
                if np.all(over <= 0): break
                clipped = np.maximum(x - U, 0)
                x -= clipped
                t_rem = int(clipped.sum())
                if t_rem == 0: break
                free_cap = U - x
                mask = free_cap > 0
                if not np.any(mask):
                    # Nowhere to put; greedily fill under hard caps.
                    ii = np.argsort(-free_cap)
                    for i in ii:
                        if t_rem == 0: break
                        add = min(free_cap[i], t_rem)
                        x[i] += add; t_rem -= add
                    break
                p2 = p.copy(); p2[~mask] = 0.0
                s2 = p2.sum()
                p2 = (p2 / s2) if s2 > 0 else (mask.astype(float)/mask.sum())
                x += rng.multinomial(t_rem, p2, size=1)[0]
                t_rem = 0
            EV[:, j] = x
        return EV

    # -- unified allocator switch (caps always enforced) --

    def _allocate_EV(self, T_idx: np.ndarray, W6: np.ndarray, allocator: Optional[str]) -> np.ndarray:
        method = (allocator or self.allocator).lower()
        if method == "round":
            alloc = W6 * T_idx[None, :]
            return self._round_allocations_to_totals(alloc, T_idx, self.max_ev)
        elif method == "multinomial":
            return self._alloc_multinomial_capped(T_idx, W6, self.max_ev, self.rng)
        else:
            raise ValueError("allocator must be 'multinomial' or 'round'")

    # -- public sampling / marginals --

    def sample(self, M: int, *, allocator: Optional[str] = None) -> np.ndarray:
        S5 = self._sample_S5(M)
        W6 = self._stick_breaking_from_unit(S5)
        T_idx = self._sample_totals(M)
        EV_mat = self._allocate_EV(T_idx, W6, allocator)
        return EV_mat.T

    def getMarginals(self, mc_samples: int = 5000, *, allocator: Optional[str] = None) -> np.ndarray:
        nS, maxEV = self.n_stats, self.max_ev
        S5 = self._sample_S5(mc_samples)
        W6 = self._stick_breaking_from_unit(S5)
        T_idx = self._sample_totals(mc_samples)
        EV_mat = self._allocate_EV(T_idx, W6, allocator)  # (6, M)
        marginals = np.zeros((nS, maxEV + 1), dtype=float)
        rows = np.repeat(np.arange(nS), mc_samples)
        cols = EV_mat.reshape(-1)
        vals = np.full(nS * mc_samples, 1.0 / mc_samples, dtype=float)
        np.add.at(marginals, (rows, cols), vals)
        s = marginals.sum(axis=1, keepdims=True)
        np.divide(marginals, s, out=marginals, where=(s > 0))
        return marginals

    def getProb(self, EV: np.ndarray, asLog: bool = False) -> np.ndarray:
        """
        Return P(EV) under this EV_PMF, computed (in log-space) as:
            log P(EV) = log P(T) + sum_{r=1..5} log P_r(S_r)
        where T = sum(EV), W6 = EV / T (with T=0 → [0,0,0,0,0,1]),
        S = invert_stick_breaking(W6[0:5]), and each S_r is binned to the
        nearest of `w_bins` centers to read off row-wise probabilities in self.W.

        Parameters
        ----------
        EV : (6,) or (N,6) array-like of integer EVs in [0, max_ev]
        asLog : bool
            If True, return log-probabilities. If False, return probabilities.

        Returns
        -------
        out : scalar (if EV shape (6,)) or (N,) ndarray
            log-probs if asLog=True, probs otherwise.
        
        Notes
        -----
        Per-stat EV cap enforcement: Any EV value exceeding max_ev for a stat is
        treated as infeasible (probability = 0, log-prob = -inf). Similarly, totals
        exceeding max_total_ev are infeasible. This ensures strict adherence to
        game mechanics and prevents invalid states from receiving probability mass.
        """
        E = np.asarray(EV, dtype=float)
        scalar_input = False
        if E.ndim == 1:
            E = E[None, :]  # (1,6)
            scalar_input = True
        if E.shape[1] != 6:
            raise ValueError("EV must have shape (6,) or (N,6).")

        N = E.shape[0]

        # Start with -inf (log(0)) everywhere; fill in valid rows
        log_probs = np.full(N, -np.inf, dtype=float)

        # Validity checks: per-stat caps and total cap
        # Per-stat EV values exceeding max_ev are infeasible
        per_stat_ok = (E >= 0.0) & (E <= self.max_ev + 1e-9)
        per_row_ok = per_stat_ok.all(axis=1)

        T = E.sum(axis=1).astype(int)  # totals
        # Total EV values exceeding max_total_ev are infeasible
        total_ok = (T >= 0) & (T <= self.max_total_ev)

        valid = per_row_ok & total_ok
        if not np.any(valid):
            # Nothing valid: return all -inf (or zeros if asLog=False)
            if asLog:
                return log_probs[0] if scalar_input else log_probs
            out = np.zeros(N, dtype=float)
            return out[0] if scalar_input else out

        # log P(T) for valid rows
        with np.errstate(divide="ignore", invalid="ignore"):
            logT = np.full(N, -np.inf, dtype=float)
            logT[valid] = np.log(self.T[T[valid]])

        # Map valid EV rows to W6 proportions (handle T=0 → [0,0,0,0,0,1])
        E_sub = E[valid, :]                  # (K,6)
        T_sub = T[valid]                     # (K,)
        with np.errstate(divide="ignore", invalid="ignore"):
            W6 = np.divide(E_sub, T_sub[:, None], out=np.zeros_like(E_sub), where=(T_sub[:, None] > 0))
        zero_tot = (T_sub == 0)
        if np.any(zero_tot):
            W6[zero_tot] = 0.0
            W6[zero_tot, 5] = 1.0

        # Invert stick-breaking per row to S5 in [0,1]
        K = W6.shape[0]
        S5 = np.empty((K, 5), dtype=float)
        for i in range(K):
            S5[i] = self._invert_stick_breaking(W6[i])

        # Bin S5 to nearest centers in [0,1] with B bins
        B = self.w_bins
        idx = np.rint(S5 * (B - 1)).astype(int)
        np.clip(idx, 0, B - 1, out=idx)  # (K,5)

        # Extract W values at binned indices W[r, idx[i,r]] for each row r and sample i
        # W is (5, B), idx is (K, 5); we want to extract W[r, idx[i,r]] for each r in 0..4, i in 0..K-1
        K = idx.shape[0]
        W_rows = np.empty((K, 5), dtype=float)
        for r in range(5):
            W_rows[:, r] = self.W[r, idx[:, r]]
        any_zero = np.any(W_rows == 0, axis=1)  # (K,) bool

        # Sum row-wise log-probs: sum_r log W[r, idx_r]
        with np.errstate(divide="ignore", invalid="ignore"):
            row_log = np.sum(np.log(np.clip(W_rows, 1e-300, 1.0)), axis=1)  # (K,)
            row_log[any_zero] = -np.inf

        # Combine for valid rows
        log_probs[valid] = logT[valid] + row_log

        if asLog:
            return log_probs[0] if scalar_input else log_probs

        # Convert back to probability space; exp(-inf) -> 0
        probs = np.exp(log_probs)
        return probs[0] if scalar_input else probs

    # deterministic mapping (kept for tests)
    @staticmethod
    def _ev_from_W6_and_T(W6_col: np.ndarray, T: int, U: int = 252) -> np.ndarray:
        y = T * np.asarray(W6_col, dtype=float)
        x = project_capped_simplex(y, T=T, U=U)
        return integerize_with_waterfill(x, T=T, U=U)

    # -- learn PMF from samples --

    @staticmethod
    def from_samples(
        samples: "np.ndarray | list[StatBlock]",
        w_bins: int = 506,
        *,
        return_coords: bool = False,
        rng: np.random.Generator | None = None,
        allocator: str = "round",
        weights: "np.ndarray | list[float] | None" = None,
    ) -> "EV_PMF | tuple[EV_PMF, dict[str, np.ndarray]]":
        """
        Build an EV_PMF from EV samples, optionally weighted.

        Parameters
        ----------
        samples : (N,6) integer EVs or list[StatBlock]
        w_bins  : number of bins for each of the 5 stick-breaking rows
        return_coords : also return {'totals','W6','S5'} if True
        rng, allocator : forwarded to EV_PMF constructor
        weights : optional (N,) nonnegative weights; if None, uniform

        Returns
        -------
        pmf or (pmf, coords)
        """
        # --- coerce samples to (N,6) float array ---
        if isinstance(samples, list):
            ev_array = np.array(
                [[s.hp, s.atk, s.def_, s.spa, s.spd, s.spe] for s in samples],
                dtype=float
            )
        else:
            ev_array = np.asarray(samples, dtype=float)
            if ev_array.ndim != 2 or ev_array.shape[1] != 6:
                raise ValueError("samples must be (N,6) EV values or a list[StatBlock].")

        N = ev_array.shape[0]
        # --- coerce/normalize weights ---
        if weights is None:
            w = np.full(N, 1.0 / max(N, 1), dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.ndim != 1 or w.shape[0] != N:
                raise ValueError("weights must be a vector of length N.")
            w = np.clip(w, 0.0, np.inf)
            s = w.sum()
            w = (w / s) if s > 0 else np.full(N, 1.0 / max(N, 1), dtype=float)

        # Use class constant for per-stat EV cap
        max_ev = EV_PMF.MAX_EV
        max_total_ev = 2 * max_ev + 6  # 510

        # --- 1) Weighted PMF over totals T ---
        totals = ev_array.sum(axis=1).astype(int)
        totals = np.clip(totals, 0, max_total_ev)
        T_hist = np.bincount(totals, weights=w, minlength=max_total_ev + 1).astype(float)
        Ts = T_hist.sum()
        T_hist = (T_hist / Ts) if Ts > 0 else np.full(max_total_ev + 1, 1.0 / (max_total_ev + 1))

        # --- 2) Weighted stick-breaking rows via sample proportions ---
        with np.errstate(divide="ignore", invalid="ignore"):
            W6 = np.divide(ev_array, totals[:, None], out=np.zeros_like(ev_array), where=(totals[:, None] > 0))
        zero_tot = (totals == 0)
        if np.any(zero_tot):
            W6[zero_tot] = 0.0
            W6[zero_tot, 5] = 1.0

        # invert stick-breaking per sample → S5 in [0,1]
        S5 = np.empty((N, 5), dtype=float)
        for i in range(N):
            S5[i] = EV_PMF._invert_stick_breaking(W6[i])

        # bin S5 with weights
        B = w_bins
        idx = np.rint(S5 * (B - 1)).astype(int)
        np.clip(idx, 0, B - 1, out=idx)

        W_counts = np.zeros((5, B), dtype=float)
        for r in range(5):
            W_counts[r] = np.bincount(idx[:, r], weights=w, minlength=B)

        row_sums = W_counts.sum(axis=1, keepdims=True)
        W_hist = np.divide(W_counts, row_sums, out=np.zeros_like(W_counts), where=(row_sums > 0))

        pmf = EV_PMF(priorT=T_hist, priorW=W_hist, w_bins=B, rng=rng, allocator=allocator)

        if not return_coords:
            return pmf
        return pmf, {"totals": totals, "W6": W6, "S5": S5}