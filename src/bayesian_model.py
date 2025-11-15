from data_structures import *
import numpy as np

class BayesianModel:
    def __init__(self):
        pass

# Compute final stats given base stats, IVs, EVs, level, and nature.
def calc_stats(base_state: StatBlock, iv: StatBlock, ev: StatBlock, level: int, nature: Nature) -> StatBlock:
    common : StatBlock = ((2 * base_state) + iv + (ev // 4))* level / 100
    hp = common.hp + level + 10
    others = (common+5)* nature.modifier_block()
    return StatBlock(hp=hp, atk=others.atk, def_=others.def_, spa=others.spa, spd=others.spd, spe=others.spe)