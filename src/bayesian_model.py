from data_structures import *
from PMFs import EV_PMF
import numpy as np

def update_ev_pmf(prior: EV_PMF, upd: EV_PMF, mode: str = "linear") -> EV_PMF:
    """
    Update EV PMF:
      - T: discrete convolution (totals add)
      - W: barycenter of the five independent stick-breaking row pmfs,
           weighted solely by E[T] from prior and update.
    mode: "linear" or "geometric"
    """
    # ---- 1) Update T via convolution ----
    # both vectors length = max_total_ev+1
    new_T = np.convolve(prior.T, upd.T)[:prior.max_total_ev + 1]
    s = new_T.sum()
    new_T = new_T / s if s > 0 else prior.T.copy()

    # ---- 2) Update W using only expected totals as masses ----
    tvals_prior = np.arange(prior.max_total_ev + 1, dtype=float)
    tvals_upd   = np.arange(upd.max_total_ev   + 1, dtype=float)
    m_prior = float((tvals_prior * prior.T).sum())
    m_upd   = float((tvals_upd   * upd.T  ).sum())
    denom = m_prior + m_upd

    if denom <= 0:
        new_W = prior.W.copy()
    else:
        if mode == "linear":
            # Dirichlet-mean / convex barycenter per row
            new_W = (m_prior * prior.W + m_upd * upd.W) / denom
            row_sums = new_W.sum(axis=1, keepdims=True)
            new_W = np.divide(new_W, row_sums, out=new_W, where=(row_sums > 0))
        elif mode == "geometric":
            # Product-of-experts barycenter per row (sharper)
            eps = 1e-12
            logW = m_prior * np.log(prior.W + eps) + m_upd * np.log(upd.W + eps)
            new_W = np.exp(logW)
            row_sums = new_W.sum(axis=1, keepdims=True)
            new_W = np.divide(new_W, row_sums, out=new_W, where=(row_sums > 0))
        else:
            raise ValueError("mode must be 'linear' or 'geometric'")

    return EV_PMF(priorT=new_T, priorW=new_W, w_bins=prior.w_bins)

class BayesianModel:
    def __init__(self):
        pass

# Compute final stats given base stats, IVs, EVs, level, and nature.
def calc_stats(base_state: StatBlock, iv: StatBlock, ev: StatBlock, level: int, nature: Nature) -> StatBlock:
    common : StatBlock = ((2 * base_state) + iv + (ev // 4))* level / 100
    hp = common.hp + level + 10
    others = (common+5)* nature.modifier_block()
    return StatBlock(hp=hp, atk=others.atk, def_=others.def_, spa=others.spa, spd=others.spd, spe=others.spe)