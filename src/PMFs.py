# PMFs.py

import numpy as np
from data_structures import *
from scipy.optimize import minimize
from typing import Optional


# ----------------------
# Utilities
# ----------------------

def project_to_simplex(v):
    """Euclidean projection onto {w: w>=0, sum w=1} (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / (np.arange(n) + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


# ----------------------
# Capped-simplex projection (sum=T, 0<=x<=U)
# ----------------------

def project_capped_simplex(y: np.ndarray, T: int, U: int = 252, tol: float = 1e-9) -> np.ndarray:
    """
    Project y in R^6 onto { x: sum x = T, 0 <= x_i <= U } in L2.
    Returns float vector x (not rounded to ints).
    """
    y = np.asarray(y, dtype=float)
    Uvec = np.full_like(y, float(U))
    # Clamp T to a valid range (defensive)
    T = float(np.clip(T, 0.0, Uvec.sum()))

    # Bisection for tau such that sum clip(y - tau, 0, U) = T
    lo = (y - Uvec).min() - 1.0
    hi = y.max() + 1.0
    x = None
    for _ in range(60):  # enough for double precision
        tau = 0.5 * (lo + hi)
        x = np.clip(y - tau, 0.0, U)
        s = x.sum()
        if abs(s - T) <= tol:
            break
        if s > T:
            lo = tau
        else:
            hi = tau
    return x if x is not None else np.clip(y, 0.0, U)


def integerize_with_waterfill(x: np.ndarray, T: int, U: int = 252) -> np.ndarray:
    """
    Turn a real feasible x (sum close to T, 0<=x<=U) into an integer EV vector:
    - round to nearest
    - adjust +/-1 guided by fractional parts while respecting caps
    """
    x = np.asarray(x, dtype=float)
    E = np.rint(x).astype(int)
    E = np.clip(E, 0, U)
    diff = int(T) - int(E.sum())
    if diff == 0:
        return E

    frac = x - np.floor(x)
    order = np.argsort(frac)[::-1] if diff > 0 else np.argsort(frac)
    step = 1 if diff > 0 else -1
    k = 0
    n = E.size
    while diff != 0 and k < n:
        i = order[k]
        cand = E[i] + step
        if 0 <= cand <= U:
            E[i] = cand
            diff -= step
        k += 1
    return E


# ----------------------
# IV PMF
# ----------------------

class IV_PMF:
    def __init__(self, prior: np.ndarray | None = None, rng: np.random.Generator | None = None):
        # shape (6, 32)
        self.rng = rng or np.random.default_rng()
        if prior is None:
            self.prior = np.full((6, 32), 1/32.0)
        else:
            arr = np.asarray(prior, dtype=float)
            assert arr.shape == (6, 32)
            self.prior = arr / arr.sum(axis=1, keepdims=True)

    @property
    def P(self) -> np.ndarray:
        return self.prior

    def normalize_(self) -> None:
        rs = self.prior.sum(axis=1, keepdims=True)
        np.divide(self.prior, rs, out=self.prior, where=(rs > 0))

    def sample(self, M: int) -> np.ndarray:
        """
        Returns (6, M) ints in 0..31 sampled row-wise from the PMFs.
        """
        cdf = np.cumsum(self.prior, axis=1)
        cdf[:, -1] = 1.0
        u = self.rng.random((6, M))
        idx = np.empty((6, M), dtype=int)
        for s in range(6):
            idx[s] = np.searchsorted(cdf[s], u[s], side="right")
        return np.clip(idx, 0, 31)

    def weighted_add_(self, iv_mat: np.ndarray, w: np.ndarray) -> None:
        """
        iv_mat: (6, M), values in 0..31
        w:      (M,) nonnegative weights summing to 1 (not strictly required; we normalize)
        """
        out = np.zeros_like(self.prior)
        for s in range(6):
            np.add.at(out[s], iv_mat[s], w)
        self.prior = out
        self.normalize_()

    def blend(self, other: "IV_PMF", mode="linear", alpha: float = 0.5) -> "IV_PMF":
        if mode == "linear":
            P = alpha*self.P + (1-alpha)*other.P
        elif mode == "geometric":
            eps = 1e-12
            P = np.exp(alpha*np.log(self.P+eps) + (1-alpha)*np.log(other.P+eps))
        else:
            raise ValueError("mode must be linear|geometric")
        out = IV_PMF(P, rng=self.rng)
        out.normalize_()
        return out


# ----------------------
# EV PMF using 5-parameter stick-breaking + capped-simplex projection
# ----------------------

class EV_PMF:
    def __init__(self, priorT: np.ndarray | None = None, priorW: np.ndarray | None = None,
                 w_bins: int = 506, rng: np.random.Generator | None = None):
        self.max_ev = 252
        self.n_stats = 6
        self.max_total_ev = 2 * self.max_ev + self.n_stats  # 510
        self.w_bins = w_bins
        self.rng = rng if rng is not None else np.random.default_rng()

        # --- PMF over totals T (shape: 511) ---
        if priorT is not None:
            assert priorT.shape == (self.max_total_ev + 1,)
            self.T = priorT.astype(float)
        else:
            self.T = np.full(self.max_total_ev + 1, 1.0 / (self.max_total_ev + 1), dtype=float)
        self.T /= self.T.sum()

        # --- PMFs over the five stick-breaking variables (shape: 5 x w_bins) ---
        if priorW is not None:
            assert priorW.shape == (5, self.w_bins)
            self.W = priorW.astype(float)
        else:
            self.W = np.full((5, self.w_bins), 1.0 / self.w_bins, dtype=float)

        # normalize each row independently
        row_sums = self.W.sum(axis=1, keepdims=True)
        self.W = np.divide(self.W, row_sums, out=self.W, where=(row_sums > 0))

        # fixed bin centers in [0,1]
        self.s_grid = np.linspace(0.0, 1.0, self.w_bins)

    # ---------- stick-breaking ----------

    @staticmethod
    def _stick_breaking_from_unit(S5: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        S5: (5, M) with entries in [0,1]; returns W6: (6, M) on the simplex.
        """
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
        """
        Invert stick-breaking: w1..w6 -> s1..s5 in [0,1].
        """
        S = np.empty(5, dtype=float)
        rem = 1.0
        for k in range(5):
            if rem <= eps:
                s_k = 0.0
            else:
                s_k = w[k] / rem
            S[k] = np.clip(s_k, 0.0, 1.0)
            rem *= (1.0 - S[k])
        return S

    # ---------- sample S ~ independent row PMFs over s_grid ----------

    def _sample_S5(self, M: int) -> np.ndarray:
        """
        Draw M i.i.d. samples of S = (s1..s5) from independent discrete pmfs in self.W.
        Returns S5 of shape (5, M) with values in [0,1] from self.s_grid.
        """
        B = self.w_bins
        row_sums = self.W.sum(axis=1, keepdims=True)
        W_row = np.divide(self.W, row_sums, out=np.zeros_like(self.W), where=(row_sums > 0))
        cdfs = np.cumsum(W_row, axis=1)
        cdfs[:, -1] = 1.0

        u = self.rng.random((5, M))
        idx = np.empty((5, M), dtype=int)
        for r in range(5):
            idx[r] = np.searchsorted(cdfs[r], u[r], side='right')
        idx = np.minimum(idx, B - 1)
        return self.s_grid[idx]

    # ---------- NEW: allocation from (T, W6 column) via capped-simplex ----------

    @staticmethod
    def _ev_from_W6_and_T(W6_col: np.ndarray, T: int, U: int = 252) -> np.ndarray:
        """
        Given a 6-way proportion W6_col (sum to 1) and a total T,
        project y = T * W6_col onto the capped simplex (sum=T, 0<=x<=U), then integerize.
        Returns int vector of length 6, sum=T, each in [0,U].
        """
        y = T * np.asarray(W6_col, dtype=float)
        x = project_capped_simplex(y, T=T, U=U)
        return integerize_with_waterfill(x, T=T, U=U)

    # ---------- marginals via Monte Carlo over S with capped-simplex projection ----------

    def getMarginals(self, mc_samples: int = 5000) -> np.ndarray:
        """
        Returns marginals[stat, ev] = P(stat has EV=ev). Shape: (6, max_ev+1).

        This version draws S~W (stick-breaking), forms 6-way proportions W6,
        then for each total T, produces EV allocations by projecting T*W6
        onto the capped simplex (sum=T, 0<=E<=U) and rounding with a
        small water-fill. This covers the full feasible region (no corner
        matrix needed).
        """
        nS, maxEV, maxT = self.n_stats, self.max_ev, self.max_total_ev

        # Draw S samples once, build W6 via stick-breaking
        S5 = self._sample_S5(mc_samples)                   # (5, M)
        W6 = self._stick_breaking_from_unit(S5)            # (6, M)

        marginals = np.zeros((nS, maxEV + 1), dtype=float)

        # Loop over totals; for each T accumulate contributions from all samples
        for T in range(maxT + 1):
            if self.T[T] == 0.0:
                continue
            w_T = self.T[T] / mc_samples  # weight per sample for this T
            # Produce EV for each column j
            for j in range(mc_samples):
                EV = self._ev_from_W6_and_T(W6[:, j], T, U=maxEV)
                # scatter-add
                for s in range(nS):
                    ev_val = EV[s]
                    marginals[s, ev_val] += w_T

        # Normalize per-stat for numerical safety
        s = marginals.sum(axis=1, keepdims=True)
        np.divide(marginals, s, out=marginals, where=(s > 0))
        return marginals
