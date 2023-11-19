import itertools
import sympy


class Operator:
    """Representation of a pauli operator (currently restricted to i, x and z).

    TODO is the restriction to positive signs ok?
    """
    length: int
    x: list[int]
    z: list[int]

    def __init__(self, length: int, *, x_positions: list[int] = None, z_positions: list[int] = None) -> None:
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

    def __repr__(self) -> str:
        ret = ["I" for _ in range(self.length)]
        for pos in self.x:
            ret[pos - 1] = "X"
        for pos in self.z:
            ret[pos - 1] = "Z"
        return "".join(ret)

    def __len__(self) -> int:
        return self.length

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


def get_check_matrix(generators: list[Operator]) -> sympy.Matrix:
    """Calculate the check matrix for the given stabilizer generators."""
    if any(len(generator) != len(generators[0]) for generator in generators):
        raise ValueError("All generators must have the same size.")
    rows = [
        [1 if pos in generator.x else 0 for pos in range(1, generator.length+1)]
        + [1 if pos in generator.z else 0 for pos in range(1, generator.length+1)]
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


def check_independent_commuting(stabilizers: list[Operator]) -> None:
    # stabilizer subspace is only non-trivial if all generators commute
    not_commuting_pairs = []
    for stab1, stab2 in itertools.combinations(stabilizers, 2):
        if not stab1.commutes(stab2):
            not_commuting_pairs.append((stab1, stab2))
    if not_commuting_pairs:
        raise ValueError(f"Following stabilizers do not commute:\n{not_commuting_pairs}")
    if not are_independent(stabilizers):
        raise ValueError("The set of stabilizers are not independent.")


def check_xj(x_j: Operator, z_j: Operator, other_z: list[Operator], stabilizers: list[Operator]) -> None:
    """Check if logical x_j fulfills the commutation relations."""
    non_commuting_stabilizers = []
    for stabilizer in stabilizers:
        if not x_j.commutes(stabilizer):
            non_commuting_stabilizers.append(stabilizer)
    if non_commuting_stabilizers:
        raise ValueError(f"x_j does not commute with the following stabilizers:\n{non_commuting_stabilizers}")
    non_commuting_z = []
    for z in other_z:
        if not x_j.commutes(z):
            non_commuting_z.append(z)
    if non_commuting_z:
        raise ValueError(f"x_j does not commute with the following z operators:\n{non_commuting_z}")
    if not x_j.anticommutes(z_j):
        raise ValueError(f"x_j does not anticommute with z_j")


stabilizers = [
    Operator(12, x_positions=[1, 4, 5, 7, 8, 10]),
    Operator(12, x_positions=[2, 3, 6, 9, 11, 12]),
    Operator(12, x_positions=[2, 5, 6, 8, 9, 11]),
    Operator(12, x_positions=[3, 4, 6, 7, 9, 12]),
    Operator(12, z_positions=[1, 4, 5, 7, 8, 10]),
    Operator(12, z_positions=[2, 3, 6, 9, 11, 12]),
    Operator(12, z_positions=[2, 5, 6, 8, 9, 11]),
    Operator(12, z_positions=[3, 4, 6, 7, 9, 12]),
]

print("Checking independency and commuting of stabilizers.")
check_independent_commuting(stabilizers)

z_1 = Operator(12, z_positions=[4, 7])
z_2 = Operator(12, z_positions=[5, 8])
z_3 = Operator(12, z_positions=[1, 5, 9, 12])
z_4 = Operator(12, z_positions=[2, 6, 7, 10])

print("Checking if logical z operators form with stabilizers an independent and commuting set.")
check_independent_commuting(stabilizers + [z_1, z_2, z_3, z_4])

x_1 = Operator(12, x_positions=[1, 4, 9, 11])
check_xj(x_1, z_1, [z_2, z_3, z_4], stabilizers)
x_2 = Operator(12, x_positions=[3, 6, 8, 10])
check_xj(x_2, z_2, [z_1, z_3, z_4], stabilizers)
x_3 = Operator(12, x_positions=[5, 8])
check_xj(x_3, z_3, [z_1, z_2, z_4], stabilizers)
x_4 = Operator(12, x_positions=[4, 7])
check_xj(x_4, z_4, [z_1, z_2, z_3], stabilizers)
