import enum
import itertools
from functools import cache
from typing import Optional

import numpy as np

SUBSCRIPT_NUMBER_MAP = {
    0: "₀",
    1: "₁",
    2: "₂",
    3: "₃",
    4: "₄",
    5: "₅",
    6: "₆",
    7: "₇",
    8: "₈",
    9: "₉",
}


def subscript_number(num: int) -> str:
    """Express a number as subscript."""
    ret = ""
    while num:
        ret = SUBSCRIPT_NUMBER_MAP[num % 10] + ret
        num = num // 10
    return ret


class Operator:
    """Representation of a pauli operator (currently restricted to i, x and z).

    TODO is the restriction to positive signs ok?
    """

    length: int
    x: list[int]
    z: list[int]
    name: str

    def __init__(
        self,
        length: int,
        *,
        x_positions: list[int] = None,
        z_positions: list[int] = None,
        name: str = None,
    ) -> None:
        x_positions = sorted(x_positions or [])
        z_positions = sorted(z_positions or [])
        if len(x_positions) != len(set(x_positions)):
            raise ValueError("Indexes in x_positions are not unique.")
        if len(z_positions) != len(set(z_positions)):
            raise ValueError("Indexes in z_positions are not unique.")
        if overlap := set(x_positions) & set(z_positions):
            raise ValueError(f"Overlap detected: {overlap}")
        for positions in [x_positions, z_positions]:
            if positions:
                if positions[-1] > length:
                    raise ValueError("Length must be >= to each position")
                if positions[0] == 0:
                    raise ValueError("Indexing starts at 1.")
        self.length = length
        self.x = x_positions
        self.z = z_positions
        self.name = name or ""

    def __repr__(self) -> str:
        ret = []
        for pos in range(1, self.length + 1):
            if pos in self.x:
                ret.append(f"X{subscript_number(pos)}")
            if pos in self.z:
                ret.append(f"Z{subscript_number(pos)}")
        if self.name:
            ret.append(f"('{self.name}')")
        return "".join(ret)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operator):
            raise NotImplementedError
        return self.x == other.x and self.z == other.z and self.length == other.length

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Operator):
            raise NotImplementedError
        if self.length != other.length:
            raise ValueError
        return self.x < other.x and self.z < other.z

    def __len__(self) -> int:
        return self.length

    def __hash__(self):
        return hash((self.length, tuple(self.x), tuple(self.z)))

    @cache
    def commutes(self, other: "Operator") -> bool:
        """Check whether two pauli operators commute or not."""
        if not isinstance(other, Operator):
            raise ValueError(f"Operator expected, got {other}.")
        if self.length != other.length:
            raise ValueError("Operators need to be of same length.")
        overlap = set(self.x) & set(other.z) | set(self.z) & set(other.x)
        return not bool(len(overlap) % 2)

    def anticommutes(self, other: "Operator") -> bool:
        """Check whether two pauli operators anticommute or not."""
        # this holds as long as we look only at I, X and Z
        return not self.commutes(other)

    @property
    def qubits(self) -> list[int]:
        """The qubits this stabilizer has support on."""
        return self.x + self.z


class Color(enum.IntEnum):
    # monochrome colors
    red = 0
    blue = 1
    green = 2
    yellow = 3

    # mixed colors
    rb = 4
    rg = 5
    ry = 6
    bg = 7
    by = 8
    gy = 9

    @classmethod
    def get_monochrome(cls) -> list["Color"]:
        return [Color.red, Color.blue, Color.green, Color.yellow]

    @property
    def is_monochrome(self) -> bool:
        """Is this color a pure color?"""
        return self in self.get_monochrome()

    @classmethod
    def get_mixed(cls) -> list["Color"]:
        return [Color.rb, Color.rg, Color.ry, Color.bg, Color.by, Color.gy]

    @property
    def is_mixed(self) -> bool:
        return self in self.get_mixed()

    @property
    def as_names(self) -> list[str]:
        """Return the ingredients of the color, f.e. Color.rb -> ['red', 'blue']."""
        return {
            Color.red: ["red"],
            Color.blue: ["lightblue"],
            Color.green: ["green"],
            Color.yellow: ["yellow"],
            # for better visibility
            Color.rb: ["lightblue", "red"],
            Color.rg: ["red", "green"],
            Color.ry: ["red", "yellow"],
            Color.bg: ["lightblue", "green"],
            Color.by: ["lightblue", "yellow"],
            Color.gy: ["green", "yellow"],
        }[self]

    @classmethod
    def color_map(cls) -> list[str]:
        return [
            "red",          # red
            "lightblue",    # blue
            "darkgreen",    # green
            "gold",         # yellow
            "magenta",      # red blue
            "brown",        # red green
            "orange",       # red yellow
            "blue",         # blue green
            "grey",         # blue yellow
            "lightgreen"    # green yellow
        ]

    @classmethod
    def color_limits(cls) -> list[int]:
        return [min(cls), max(cls)]

    def combine(self, other: "Color") -> "Color":
        if not (self.is_monochrome and other.is_monochrome):
            raise ValueError
        # only for debug purpose
        if self == other:
            return self
        if self > other:
            key = (other, self)
        else:
            key = (self, other)
        return {
            (Color.red, Color.blue): Color.rb,
            (Color.red, Color.green): Color.rg,
            (Color.red, Color.yellow): Color.ry,
            (Color.blue, Color.green): Color.bg,
            (Color.blue, Color.yellow): Color.by,
            (Color.green, Color.yellow): Color.gy,
        }[key]

    def intersect(self, other: "Color") -> Optional["Color"]:
        if not (self.is_mixed and other.is_mixed):
            raise ValueError
        for mono in self.get_monochrome():
            if self.contains(mono) and other.contains(mono):
                return mono
        return None

    def contains(self, other: "Color") -> bool:
        if not (self.is_mixed and other.is_monochrome):
            raise ValueError
        for mono in self.get_monochrome():
            if mono == other:
                continue
            if mono.combine(other) == self:
                return True
        return False

    @property
    def components(self) -> list["Color"]:
        if self.is_monochrome:
            return [self]
        return {
            Color.rb: [Color.red, Color.blue],
            Color.rg: [Color.red, Color.green],
            Color.ry: [Color.red, Color.yellow],
            Color.bg: [Color.blue, Color.green],
            Color.by: [Color.blue, Color.yellow],
            Color.gy: [Color.green, Color.yellow],
        }[self]



class Stabilizer(Operator):
    """Special kind of operator which is viewed as a color code stabilizer."""

    ancillas: list[int]
    color: Color

    def __init__(
        self,
        length: int,
        color: Color,
        *,
        x_positions: list[int] = None,
        z_positions: list[int] = None,
        name: str = None,
        ancillas: list[int] = None,
    ) -> None:
        super().__init__(
            length, x_positions=x_positions, z_positions=z_positions, name=name
        )
        # color code stabilizers have either only x or only z support
        if self.x and self.z:
            raise RuntimeError(f"All stabilizers need to be x or z type: {self}")

        # check that ancilla qubits do not overlap with data qubits
        ancillas = sorted(ancillas or [])
        if overlap := set(ancillas) & set(self.x):
            raise ValueError(f"Overlap between ancillas and x_positions: {overlap}")
        if overlap := set(ancillas) & set(self.z):
            raise ValueError(f"Overlap between ancillas and z_positions: {overlap}")
        self.ancillas = ancillas
        self.color = color


def get_check_matrix(generators: list[Operator], only_x: bool = False, only_z: bool = False) -> np.matrix:
    """Calculate the check matrix for the given stabilizer generators."""
    if any(len(generator) != len(generators[0]) for generator in generators):
        raise ValueError("All generators must have the same size.")
    if only_x and only_z:
        raise ValueError
    if only_x:
        rows = [
            [1 if pos in generator.x else 0 for pos in range(1, generator.length + 1)]
            for generator in generators if generator.x
        ]
    elif only_z:
        rows = [
            [1 if pos in generator.z else 0 for pos in range(1, generator.length + 1)]
            for generator in generators if generator.z
        ]
    else:
        rows = [
            [1 if pos in generator.x else 0 for pos in range(1, generator.length + 1)]
            + [1 if pos in generator.z else 0 for pos in range(1, generator.length + 1)]
            for generator in generators
        ]
    return np.matrix(rows)


def are_independent(stabilizers: list[Operator]) -> bool:
    """Check if the given stabilizers set is independent.

    The set stabilizers of stabilizers are independent iff the rows of
    their check matrix are linear independent.
    """
    return len(stabilizers) == count_independent(stabilizers)


def count_independent(stabilizers: list[Operator]) -> int:
    """Calculate the number of independent stabilizers.

    This is the rank of their check matrix.
    """
    check_stabilizers(stabilizers, check_independence=False)
    # since x and z stabilizers are always independent, we count them separately to reduce overhead
    x_matrix = get_check_matrix(stabilizers, only_x=True)
    x_rank = 0
    if x_matrix.any():
        x_rank = np.linalg.matrix_rank(x_matrix)
    z_matrix = get_check_matrix(stabilizers, only_z=True)
    z_rank = 0
    if z_matrix.any():
        z_rank = np.linalg.matrix_rank(z_matrix)
    return x_rank + z_rank


def check_stabilizers(stabilizers: list[Operator], check_independence: bool = True) -> None:
    """Check if a set of operators form a set of stabilizers.

    This check is necessary and sufficient.
    """
    # stabilizer subspace is only non-trivial if all generators commute
    not_commuting_pairs = []
    for stab1, stab2 in itertools.combinations(stabilizers, 2):
        if not stab1.commutes(stab2):
            not_commuting_pairs.append((stab1, stab2))
    if not_commuting_pairs:
        raise ValueError(
            f"Following stabilizers do not commute:\n{not_commuting_pairs}"
        )
    if check_independence and not are_independent(stabilizers):
        raise ValueError("The set of stabilizers are not independent.")


def check_logical_operator(logical: Operator, stabilizers: list[Operator]) -> None:
    """Check if a given operator is indeed a logical operator for this stabilizer code.

    This checks only necessary conditions, but is not sufficient.
    It needs to be checked that all logical operators and the stabilizers form an
    independent set, and that the logical operators perform the correct operation.
    """
    non_commuting_stabilizers = []
    for stabilizer in stabilizers:
        if not logical.commutes(stabilizer):
            non_commuting_stabilizers.append(stabilizer)
    if non_commuting_stabilizers:
        raise ValueError(
            f"{logical} does not commute with the following stabilizers:\n{non_commuting_stabilizers}"
        )
    if count_independent(stabilizers + [logical]) != count_independent(stabilizers) + 1:
        raise ValueError(
            f"{logical} does not form an independent set with the stabilizers."
        )


def check_logical_operators(
    logicals: list[Operator], stabilizers: list[Operator]
) -> None:
    """Check if a given list of operators are logical operators for this stabilizer code.

    This checks only necessary conditions, but is not sufficient.
    It needs to be checked that all logical operators perform the correct operation.
    """
    for logical in logicals:
        check_logical_operator(logical, stabilizers)
    if count_independent(stabilizers + logicals) != count_independent(stabilizers) + len(logicals):
        raise ValueError("The logical operators do not form an independent set.")


def check_z(z: list[Operator], stabilizers: list[Operator]) -> None:
    """Check if logical z operators fulfill the commutation relations.

    This is necessary and sufficient.
    """
    check_logical_operators(z, stabilizers)
    not_commuting_pairs = []
    for z1, z2 in itertools.combinations(z, 2):
        if not z1.commutes(z2):
            not_commuting_pairs.append((z1, z2))
    if not_commuting_pairs:
        raise ValueError(
            f"Following z operators do not commute:\n{not_commuting_pairs}"
        )


def check_xj(
    x_j: Operator, z_j: Operator, other_z: list[Operator], stabilizers: list[Operator]
) -> None:
    """Check if logical x_j fulfills the commutation relations.

    This is necessary and sufficient.
    """
    check_logical_operator(x_j, stabilizers)
    non_commuting_z = []
    for z in other_z:
        if not x_j.commutes(z):
            non_commuting_z.append(z)
    if non_commuting_z:
        raise ValueError(
            f"{x_j} does not commute with the following z operators:\n{non_commuting_z}"
        )
    if not x_j.anticommutes(z_j):
        raise ValueError(f"{x_j} does not anticommute with {z_j}")
