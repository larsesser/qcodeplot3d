import enum
import itertools

import sympy

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
    red = 1
    blue = 2
    green = 3
    yellow = 4

    # mixed colors
    rb = 11
    rg = 12
    ry = 13
    bg = 14
    by = 15
    gy = 16

    @property
    def is_monochrome(self) -> bool:
        """Is this color a pure color?"""
        return self in {Color.red, Color.blue, Color.green, Color.yellow}

    @property
    def is_mixed(self) -> bool:
        return self in {Color.rb, Color.rg, Color.ry, Color.bg, Color.by, Color.gy}

    @property
    def as_names(self) -> list[str]:
        """Return the ingredients of the color, f.e. Color.rb -> ['red', 'blue']."""
        return {
            Color.red: ["red"],
            Color.blue: ["blue"],
            Color.green: ["green"],
            Color.yellow: ["yellow"],
            # for better visibility
            Color.rb: ["blue", "red"],
            Color.rg: ["red", "green"],
            Color.ry: ["red", "yellow"],
            Color.bg: ["blue", "green"],
            Color.by: ["blue", "yellow"],
            Color.gy: ["green", "yellow"],
        }[self]

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


def get_check_matrix(generators: list[Operator]) -> sympy.Matrix:
    """Calculate the check matrix for the given stabilizer generators."""
    if any(len(generator) != len(generators[0]) for generator in generators):
        raise ValueError("All generators must have the same size.")
    rows = [
        [1 if pos in generator.x else 0 for pos in range(1, generator.length + 1)]
        + [1 if pos in generator.z else 0 for pos in range(1, generator.length + 1)]
        for generator in generators
    ]
    return sympy.Matrix(rows)


def are_independent(stabilizers: list[Operator]) -> bool:
    """Check if the given stabilizers set is independent.

    The set stabilizers of stabilizers are independent iff the rows of
    their check matrix are linear independent.
    """
    check_matrix = get_check_matrix(stabilizers)
    # perform gaussian elimination by calculating the reduced row-echelon form of the matrix
    rref = check_matrix.rref(pivots=False)
    assert isinstance(rref, sympy.Matrix)
    # the rows are linear independent if there is no row with only zeros
    # rows are sorted, so checking the last row is sufficient
    return any(e != 0 for e in rref.row(-1))


def count_independent(stabilizers: list[Operator]) -> int:
    """Calculate the number of independent stabilizers.

    This is the rank of their check matrix.
    """
    check_matrix = get_check_matrix(stabilizers)
    return check_matrix.rank()


def check_stabilizers(stabilizers: list[Operator]) -> None:
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
    if not are_independent(stabilizers):
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
    if not are_independent(stabilizers + [logical]):
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
    if not are_independent(stabilizers + logicals):
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
