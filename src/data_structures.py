from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Iterable, Iterator, Optional
import bisect

# =====================
# Core stat structures
# =====================

class StatType(Enum):
    HP = 0
    ATTACK = 1
    DEFENSE = 2
    SPECIAL_ATTACK = 3
    SPECIAL_DEFENSE = 4
    SPEED = 5


@dataclass
class StatBlock:
    """
    Generic 6-stat container.
    Used for base stats, IV guesses, EV guesses, etc.
    """
    hp: int = 0
    atk: int = 0
    def_: int = 0
    spa: int = 0
    spd: int = 0
    spe: int = 0

    def __getitem__(self, key: StatType) -> int:
        if not isinstance(key, StatType):
            raise TypeError("StatBlock indices must be StatType")
        attr = key.name.lower()
        return getattr(self, attr)

    def __setitem__(self, key: StatType, value: int) -> None:
        if not isinstance(key, StatType):
            raise TypeError("StatBlock indices must be StatType")
        attr = key.name.lower()
        setattr(self, attr, value)

    def __add__(self, other: "StatBlock") -> "StatBlock":
        return StatBlock(
            hp=self.hp + other.hp,
            atk=self.atk + other.atk,
            def_=self.def_ + other.def_,
            spa=self.spa + other.spa,
            spd=self.spd + other.spd,
            spe=self.spe + other.spe,
        )

    def __sub__(self, other: "StatBlock") -> "StatBlock":
        """
        Element-wise subtraction (may be negative).
        Use clamp_nonnegative() on the result if you need only >=0.
        """
        return StatBlock(
            hp=self.hp - other.hp,
            atk=self.atk - other.atk,
            def_=self.def_ - other.def_,
            spa=self.spa - other.spa,
            spd=self.spd - other.spd,
            spe=self.spe - other.spe,
        )

    def clamp_nonnegative(self) -> "StatBlock":
        return StatBlock(
            hp=max(0, self.hp),
            atk=max(0, self.atk),
            def_=max(0, self.def_),
            spa=max(0, self.spa),
            spd=max(0, self.spd),
            spe=max(0, self.spe),
        )

    def to_dict(self) -> dict:
        return {
            "hp": self.hp,
            "attack": self.atk,
            "defense": self.def_,
            "special_attack": self.spa,
            "special_defense": self.spd,
            "speed": self.speed,
        }

    @classmethod
    def zeros(cls) -> "StatBlock":
        return cls()


# =====================
# Natures
# =====================

@dataclass(frozen=True)
class Nature:
    """
    A nature modifier.
    Known to the player; used as input to the solver.
    """
    name: str
    inc: Optional[StatType]
    dec: Optional[StatType]

    def modifier(self, stat: StatType) -> float:
        if self.inc == stat:
            return 1.1
        if self.dec == stat:
            return 0.9
        return 1.0

# =====================
# Growth rates
# =====================

class GrowthRate(Enum):
    MEDIUM_FAST = 0
    ERRATIC = 1
    FLUCTUATING = 2
    MEDIUM_SLOW = 3
    FAST = 4
    SLOW = 5


# =====================
# Species-level info
# =====================

@dataclass(frozen=True)
class SpeciesInfo:
    """
    Permanent, public information for a species:
      - base_stats
      - growth_rate
      - base_exp_yield
      - ev_yield (what this species grants when KO'd)
    This is the "a priori" species config the solver relies on.
    """
    name: str
    base_stats: StatBlock
    growth_rate: GrowthRate
    base_exp_yield: int
    ev_yield: StatBlock
    is_trainer_owned: bool = False

    def exp_to_level(self, level: int) -> int:
        """
        Total EXP required to reach `level` from level 1 for this growth rate.
        Matches standard formulas used in gens 3–5.
        """
        if level <= 1:
            return 0

        if self.growth_rate == GrowthRate.MEDIUM_FAST:
            return level**3

        if self.growth_rate == GrowthRate.ERRATIC:
            # Standard Erratic formula
            if level <= 50:
                return int(level**3 * (100 - level) / 50)
            if level <= 68:
                return int(level**3 * (150 - level) / 100)
            if level <= 98:
                return int(level**3 * (1911 - 10 * level) / 500)
            return int(level**3 * (160 - level) / 100)

        if self.growth_rate == GrowthRate.FLUCTUATING:
            if level <= 15:
                return int(level**3 * ((level + 1) / 3 + 24) / 50)
            if level <= 36:
                return int(level**3 * (level + 14) / 50)
            return int(level**3 * ((level / 2) + 32) / 50)

        if self.growth_rate == GrowthRate.MEDIUM_SLOW:
            # 6/5 n^3 - 15 n^2 + 100 n - 140
            return int((6 / 5) * level**3 - 15 * level**2 + 100 * level - 140)

        if self.growth_rate == GrowthRate.FAST:
            return int(4 * level**3 / 5)

        if self.growth_rate == GrowthRate.SLOW:
            return int(5 * level**3 / 4)

        raise ValueError("Unknown growth rate")
    
    def exp_bin_search(self, exp: int, low: int = 1, high: int = 101) -> int:
        """
        Given an EXP amount, find the corresponding level using binary search.
        Assumes levels are in [low, high).
        """
        if (high == low + 1):
            return low
        mid = (low + high) // 2
        mid_exp = self.exp_to_level(mid)
        if mid_exp < exp:
            return self.exp_bin_search(exp, mid, high)
        elif mid_exp > exp:
            return self.exp_bin_search(exp, low, mid)
        else:
            return mid

    def level_from_exp(self, exp: int) -> int:
        """
        Given an EXP amount, find the corresponding level.
        Uses binary search over levels 1–100.
        """
        return self.exp_bin_search(exp)

    def get_exp_yield_flat(
        self,
        fainted_level: int,
        gen: int,
    ) -> int:
        """
        Flat EXP formula:
          - Gen 1–4: floor(a * b * L / 7)
          - Gen 6:   floor(a * b * L / 5)
        where:
          a = 1.5 if fainted mon is trainer-owned, else 1.0
          b = base_exp_yield
          L = fainted_level

        Assumes:
          - single recipient
          - no Exp Share / Lucky Egg / traded / etc.
        """
        a = 1.5 if self.is_trainer_owned else 1.0
        numer = a * fainted_level * self.base_exp_yield

        if gen == 6:
            return int(numer / 5)
        return int(numer / 7)

    def get_exp_yield_scaled(
        self,
        fainted_level: int,
        winner_level: int,
        gen: int,
    ) -> int:
        """
        Scaled EXP formula skeleton:
          - Gen 5:  floor(base * F) + 1
          - Gen 7+: floor(base * F)
        with:
          base = floor(a * b * L / 5)
          F    = ((2L + 10) / (L + Lp + 10))^2.5

        Assumes:
          - single recipient
          - no Exp Share / Lucky Egg / traded / etc.
        """
        a = 1.5 if self.is_trainer_owned else 1.0
        base = int(a * fainted_level * self.base_exp_yield / 5)

        X = 2 * fainted_level + 10
        Y = fainted_level + winner_level + 10
        F = (X / Y) ** 2.5

        total = int(base * F)

        if gen == 5:
            total += 1

        return total

    def get_exp_yield(
        self,
        fainted_level: int,
        winner_level: int,
        gen: int,
    ) -> int:
        """
        Unified EXP entry point for this simplified model.

        - Gen 5 and Gen 7+ → scaled
        - Gen 1–4 and Gen 6 → flat
        """
        if gen == 5 or gen >= 7:
            return self.get_exp_yield_scaled(
                fainted_level=fainted_level,
                winner_level=winner_level,
                gen=gen
            )
        else:
            return self.get_exp_yield_flat(
                fainted_level=fainted_level,
                gen=gen
            )

# =====================
# Encounters & Training
# =====================

@dataclass
class EncounterOption:
    """
    One possible encounter within a TrainingBlock.
    `weight` is its relative encounter chance (doesn't need to be normalized).
    """
    target: SpeciesInfo
    weight: float
    levels: List[int]

@dataclass
class Encounter:
    """
    An actual encounter instance, with chosen level.
    """
    target: SpeciesInfo
    level: int

@dataclass
class TrainingBlock:
    """
    A contiguous training segment with a fixed encounter table.

      - start_level, end_level: interpreted as a range of the trainee's levels
      - location: label for human readability
      - encounters: encounter options for this block
      - trainer_owned: if True, KOs here use the trainer-owned EXP multiplier (a = 1.5)

    How you *use* this in the simulator (e.g., "only KO certain species") is
    policy-layer logic on top of these definitions.
    """
    start_level: int
    end_level: int
    location: str
    encounters: List[EncounterOption]
    trainer_owned: bool = False


class TrainingRegimen:
    """
    Ordered collection of TrainingBlock objects with no gaps or overlaps.
    Enforces:
      previous.end_level == next.start_level for all neighbors.
    """

    def __init__(self, blocks: Optional[Iterable[TrainingBlock]] = None) -> None:
        self._blocks: List[TrainingBlock] = []
        if blocks:
            self._blocks = sorted(list(blocks), key=lambda b: b.start_level)
            self._validate_blocks()

    def add_block(self, block: TrainingBlock) -> None:
        if not isinstance(block.start_level, int) or not isinstance(block.end_level, int):
            raise TypeError("start_level and end_level must be integers")
        if block.start_level >= block.end_level:
            raise ValueError("block.start_level must be less than block.end_level")

        starts = [b.start_level for b in self._blocks]
        idx = bisect.bisect_left(starts, block.start_level)

        # Insert at beginning (if any existing)
        if idx == 0 and self._blocks:
            nxt = self._blocks[0]
            if block.end_level != nxt.start_level:
                raise ValueError("New first block must end where existing first block starts")

        # Insert at end (if any existing)
        if idx == len(self._blocks) and self._blocks:
            prev = self._blocks[-1]
            if prev.end_level != block.start_level:
                raise ValueError("New last block must start where existing last block ends")

        # Insert in middle
        if 0 < idx < len(self._blocks):
            prev = self._blocks[idx - 1]
            nxt = self._blocks[idx]
            if prev.end_level != block.start_level or block.end_level != nxt.start_level:
                raise ValueError("Block must connect exactly between neighbors (no gaps/overlaps)")

        self._blocks.insert(idx, block)

    def extend(self, blocks: Iterable[TrainingBlock]) -> None:
        for b in blocks:
            self.add_block(b)

    def _validate_blocks(self) -> None:
        if not self._blocks:
            return
        for i, b in enumerate(self._blocks):
            if not isinstance(b.start_level, int) or not isinstance(b.end_level, int):
                raise TypeError("start_level and end_level must be integers")
            if b.start_level >= b.end_level:
                raise ValueError("Each block must have start_level < end_level")
            if i > 0:
                prev = self._blocks[i - 1]
                if prev.end_level != b.start_level:
                    raise ValueError("Blocks must be contiguous and non-overlapping")

    def __len__(self) -> int:
        return len(self._blocks)

    def __iter__(self) -> Iterator[TrainingBlock]:
        return iter(self._blocks)

    def __getitem__(self, idx: int) -> TrainingBlock:
        return self._blocks[idx]

    def __repr__(self) -> str:
        return f"TrainingRegimen(blocks={self._blocks!r})"

    @property
    def blocks(self) -> List[TrainingBlock]:
        # return a shallow copy to prevent accidental external mutation
        return list(self._blocks)

# =====================
# Known Pokémon context
# =====================

@dataclass(frozen=True)
class ObservedStats:
    """
    Observed stats for a specific Pokémon at a specific level.
    Used as input to the solver.
    """
    level: int
    stats: StatBlock

@dataclass
class KnownPokemonContext:
    """
    All *known* inputs for a specific Pokémon, as seen by the player/solver.

    The solver then infers:
      - a distribution over IVs (StatBlock)
      - a distribution over EVs (StatBlock)
    consistent with:
      - this context
      - observed stats
      - training regimen / encounter model.
    """
    species: SpeciesInfo
    nature: Nature
    measurements: List[ObservedStats]
    regimen: TrainingRegimen