# PMFs.py

from __future__ import annotations
import numpy as np
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
        cdf = np.cumsum(self.prior, axis=1); cdf[:, -1] = 1.0
        u = self.rng.random((6, M))
        idx = np.empty((6, M), dtype=int)
        for s in range(6):
            idx[s] = np.searchsorted(cdf[s], u[s], side="right")
        return np.clip(idx, 0, 31)

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

    def __init__(
        self,
        priorT: np.ndarray | None = None,
        priorW: np.ndarray | None = None,
        w_bins: int = 506,
        rng: np.random.Generator | None = None,
        *,
        allocator: str = "round",   # 'multinomial' | 'round'
    ):
        self.max_ev = 252
        self.n_stats = 6
        self.max_total_ev = 2 * self.max_ev + self.n_stats  # 510
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
        alloc = np.maximum(alloc, 0.0)
        ev_floor = np.floor(alloc).astype(int)
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
            cap  = max_ev - out[rows, cols]
            inc  = (remaining > 0).astype(int)
            inc  = np.minimum(inc, cap)
            out[rows, cols] += inc
            remaining -= inc
        if np.any(remaining > 0):
            cols = np.where(remaining > 0)[0]
            for r in range(6):
                if cols.size == 0: break
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

    # deterministic mapping (kept for tests)
    @staticmethod
    def _ev_from_W6_and_T(W6_col: np.ndarray, T: int, U: int = 252) -> np.ndarray:
        y = T * np.asarray(W6_col, dtype=float)
        x = project_capped_simplex(y, T=T, U=U)
        return integerize_with_waterfill(x, T=T, U=U)

    # -- learn PMF from samples --

    @staticmethod
    def from_samples(
        samples: "np.ndarray | List[StatBlock]",
        w_bins: int = 506,
        *,
        return_coords: bool = False,
        rng: np.random.Generator | None = None,
        allocator: str = "round",
    ) -> "EV_PMF | Tuple[EV_PMF, Dict[str, np.ndarray]]":
        if isinstance(samples, list):
            ev_array = np.array([[s.hp, s.atk, s.def_, s.spa, s.spd, s.spe] for s in samples], dtype=float)
        else:
            ev_array = np.asarray(samples, dtype=float)
            if ev_array.ndim != 2 or ev_array.shape[1] != 6:
                raise ValueError("samples must be (N,6) EV values or a list[StatBlock].")

        N = ev_array.shape[0]; max_ev = 252; max_total_ev = 2 * max_ev + 6
        totals = ev_array.sum(axis=1).astype(int)
        totals = np.clip(totals, 0, max_total_ev)
        T_hist = np.bincount(totals, minlength=max_total_ev + 1).astype(float)
        T_hist /= max(T_hist.sum(), 1e-12)

        with np.errstate(divide="ignore", invalid="ignore"):
            W6 = np.divide(ev_array, totals[:, None], out=np.zeros_like(ev_array), where=(totals[:, None] > 0))
        zero_tot = (totals == 0)
        if np.any(zero_tot):
            W6[zero_tot] = 0.0; W6[zero_tot, 5] = 1.0

        S5 = np.empty((N, 5), dtype=float)
        for i in range(N): S5[i] = EV_PMF._invert_stick_breaking(W6[i])

        B = w_bins
        idx = np.rint(S5 * (B - 1)).astype(int); np.clip(idx, 0, B - 1, out=idx)
        W_counts = np.zeros((5, B), dtype=float)
        for r in range(5):
            W_counts[r] = np.bincount(idx[:, r], minlength=B)
        row_sums = W_counts.sum(axis=1, keepdims=True)
        W_hist = np.divide(W_counts, row_sums, out=np.zeros_like(W_counts), where=(row_sums > 0))

        pmf = EV_PMF(priorT=T_hist, priorW=W_hist, w_bins=B, rng=rng, allocator=allocator)

        if not return_coords:
            return pmf
        return pmf, {"totals": totals, "W6": W6, "S5": S5}
