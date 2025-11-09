from dataclasses import dataclass
from enum import Enum
from typing import List, Iterable, Iterator, Optional
import bisect

class StatType(Enum):
    HP = 0
    ATTACK = 1
    DEFENSE = 2
    SPECIAL_ATTACK = 3
    SPECIAL_DEFENSE = 4
    SPEED = 5

@dataclass
class StatBlock:
    hp: int = 0
    attack: int = 0
    defense: int = 0
    special_attack: int = 0
    special_defense: int = 0
    speed: int = 0

    def __getitem__(self, key: StatType) -> int:
        return {
            StatType.HP: self.hp,
            StatType.ATTACK: self.attack,
            StatType.DEFENSE: self.defense,
            StatType.SPECIAL_ATTACK: self.special_attack,
            StatType.SPECIAL_DEFENSE: self.special_defense,
            StatType.SPEED: self.speed
        }[key]
    
    def __setitem__(self, key: StatType, value: int) -> None:
        if isinstance(key, StatType):
            key = key.value if isinstance(key.value, str) else key.name.lower()
        setattr(self, key, value)

    def __add__(self, other: 'StatBlock') -> 'StatBlock':
        return StatBlock(
            hp=self.hp + other.hp,
            attack=self.attack + other.attack,
            defense=self.defense + other.defense,
            special_attack=self.special_attack + other.special_attack,
            special_defense=self.special_defense + other.special_defense,
            speed=self.speed + other.speed
        )

    def __sub__(self, other: 'StatBlock') -> 'StatBlock':
        return StatBlock(
            hp=self.hp - other.hp,
            attack=self.attack - other.attack,
            defense=self.defense - other.defense,
            special_attack=self.special_attack - other.special_attack,
            special_defense=self.special_defense - other.special_defense,
            speed=self.speed - other.speed
        ).clamp_nonnegative()
    
    def to_dict(self) -> dict:
        return {
            "hp": self.hp,
            "attack": self.attack,
            "defense": self.defense,
            "special_attack": self.special_attack,
            "special_defense": self.special_defense,
            "speed": self.speed,
        }
    
    @classmethod
    def zeros(cls) -> "StatBlock":
        return cls()

    def clamp_nonnegative(self) -> "StatBlock":
        return StatBlock(
            hp=max(0, self.hp),
            attack=max(0, self.attack),
            defense=max(0, self.defense),
            special_attack=max(0, self.special_attack),
            special_defense=max(0, self.special_defense),
            speed=max(0, self.speed),
        )

@dataclass
class TargetPokemonSpecies:
    species: str
    ev_yield: StatBlock
    base_exp_yield: int

    def getEXPYieldFlat(self, level: int, gen: int, isTrained: bool) -> int:
        a = 1.5 if isTrained else 1.0
        numer = a* level * self.base_exp_yield
        if gen == 6:
            return int(numer / 5)
        return int(numer / 7)

    def getEXPYieldScaled(self, koed_level: int, winner_level: int, gen: int, isTrained: bool) -> int:
        a = 1.5 if isTrained else 1.0
        base = int(a * koed_level * self.base_exp_yield / 5)

        X= 2*koed_level + 10
        Y= koed_level + winner_level + 10
        F = (X/Y)**2.5
        total = int(base * F)
        if gen < 6:
            total += 1
        return total
    
    def getEXPYield(self, koed_level: int, winner_level: int, gen: int, isTrained: bool) -> int:
        if gen == 5 or gen > 6:
            return self.getEXPYieldScaled(koed_level, winner_level, gen, isTrained)
        else:
            return self.getEXPYieldFlat(winner_level, gen, isTrained)

@dataclass(frozen=True)
class Nature:
    name: str
    inc: Optional[StatType]
    dec: Optional[StatType]

    def modifier(self, stat: StatType) -> float:
        if self.inc == stat:
            return 1.1
        if self.dec == stat:
            return 0.9
        return 1.0

class GrowthRate(Enum):
    MFAST = 0
    Erratic = 1
    Fluctuating = 2
    MSLOW = 3
    FAST = 4
    SLOW = 5

@dataclass
class PlayerPokemonSpecies:
    species: str
    base_stats: StatBlock
    nature: Nature
    growth_rate: GrowthRate

    def get_level_exp(self, level: int) -> int:
        if self.growth_rate == GrowthRate.MFAST:
            return level**3
        elif self.growth_rate == GrowthRate.Erratic:
            if level <= 50:
                return int((level**3 * (100 - level)) / 50)
            elif level <= 68:
                return int((level**3 * (150 - level)) / 100)
            elif level <= 98:
                return int((level**3 * int((1911 - 10 * level) / 3)) / 500)
            else:
                return int((level**3 * (160 - level)) / 100)
        elif self.growth_rate == GrowthRate.Fluctuating:
            if level <= 15:
                return int(level**3 * (int((level + 1) / 3) + 24) / 50)
            elif level <= 36:
                return int(level**3 * (level + 14) / 50)
            else:
                return int(level**3 * (int(level / 2) + 32) / 50)
        elif self.growth_rate == GrowthRate.MSLOW:
            return int((6/5) * level**3 - 15 * level**2 + 100 * level - 140)
        elif self.growth_rate == GrowthRate.FAST:
            return int(4 * level**3 / 5)
        elif self.growth_rate == GrowthRate.SLOW:
            return int(5 * level**3 / 4)
        else:
            raise ValueError("Unknown growth rate")

@dataclass
class EncounterOption:
    target: TargetPokemonSpecies
    weight: float  # relative encounter rate on that route
    level_min: int
    level_max: int

@dataclass
class TrainingBlock:
    start_level: int
    end_level: int
    location: str
    encounters: List[EncounterOption]
    isTrained: bool = False

class TrainingRegimen:
    """
    Ordered collection of TrainingBlock objects with no gaps or overlaps.
    Consecutive blocks must satisfy previous.end_level == next.start_level.
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

        # Ensure it doesn't conflict with previous block
        if idx > 0:
            prev = self._blocks[idx - 1]
            if prev.end_level != block.start_level:
                raise ValueError(
                    "Block does not connect to previous block (gap or overlap detected)"
                )

        # Ensure it doesn't conflict with next block
        if idx < len(self._blocks):
            nxt = self._blocks[idx]
            if block.end_level != nxt.start_level:
                raise ValueError(
                    "Block does not connect to next block (gap or overlap detected)"
                )

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
        # return a shallow copy to avoid external mutation
        return list(self._blocks)
