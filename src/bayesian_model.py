from data_structures import *
from PMFs import EV_PMF
import numpy as np

def update_ev_pmf(prior_pmf: EV_PMF, update: EV_PMF) -> EV_PMF:
    """
    prior_pmf: EV_PMF representing the prior distribution.
    update: EV_PMF representing the EV gain likelihood between intervals.

    Returns the updated EV_PMF.
    """
    # Pi+1(x) = Pi(x-u)*P(u) summed over u
    new_T = np.zeros_like(prior_pmf.T)
    for u in range(update.max_total_ev + 1):
        shifted_T = np.roll(prior_pmf.T, u)
        shifted_T[:u] = 0  # zero out invalid wrap-around
        new_T += shifted_T * update.T[u]
    new_T /= new_T.sum()  # normalize

    # as W is conditionally independent of T, 
    # take the weighted sum over all T values
    prior_w = np.zeros_like(prior_pmf.W)
    for t_old in range(prior_pmf.max_total_ev + 1):
        prior_w += t_old*prior_pmf.T[t_old]

    update_w = np.zeros_like(update.W)
    for t_new in range(update.max_total_ev + 1):
        update_w += t_new*update.T[t_new]

    new_W = (prior_w*prior_pmf.W + update_w*update.W) / (prior_w + update_w)

    return EV_PMF(priorT=new_T, priorW=new_W)

class BayesianModel:
    def __init__(self):
        pass

# Compute final stats given base stats, IVs, EVs, level, and nature.
def calc_stats(base_state: StatBlock, iv: StatBlock, ev: StatBlock, level: int, nature: Nature) -> StatBlock:
    common : StatBlock = ((2 * base_state) + iv + (ev // 4))* level / 100
    hp = common.hp + level + 10
    others = (common+5)* nature.modifier_block()
    return StatBlock(hp=hp, atk=others.atk, def_=others.def_, spa=others.spa, spd=others.spd, spe=others.spe)