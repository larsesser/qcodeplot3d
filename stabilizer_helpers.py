import itertools
import sympy

Operator = list[str]

i = "i"
x = "x"
z = "z"


def get_stabilizer(length: int, *, x_positions: list[int] = None, z_positions: list[int] = None) -> Operator:
    """Return a stabilizer with given length and x/z operators at given positions.

    Positions are 1-indexed.
    """
    x_positions = x_positions or []
    z_positions = z_positions or []
    ret = [i for _ in range(length)]
    if len(x_positions) != len(set(x_positions)):
        raise ValueError("Indexes in x_positions are not unique.")
    if len(z_positions) != len(set(z_positions)):
        raise ValueError("Indexes in z_positions are not unique.")
    if overlap := set(x_positions) & set(z_positions):
        raise ValueError(f"Overlap detected: {overlap}")
    for pos in x_positions:
        ret[pos-1] = x
    for pos in z_positions:
        ret[pos-1] = z
    return ret


def is_commuting(self: Operator, other: Operator) -> bool:
    """Check whether two pauli operators commute or not."""
    if len(self) != len(other):
        raise ValueError("Operators need to be of same length.")
    overlap = 0
    for op1, op2 in zip(self, other):
        if op1 in {x, z} and op2 in {x, z}:
            if op1 != op2:
                overlap += 1
    return not bool(overlap % 2)


def is_anticommuting(self: Operator, other: Operator) -> bool:
    """Check whether two pauli operators anticommute or not."""
    # this holds as long as we look only at I, X and Z
    return not is_commuting(self, other)


def get_check_matrix(generators: list[Operator]) -> sympy.Matrix:
    """Calculate the check matrix for the given stabilizer generators."""
    if any(len(generator) != len(generators[0]) for generator in generators):
        raise ValueError("All generators must have the same size.")
    if any(e not in {i, x, z} for e in itertools.chain.from_iterable(generators)):
        raise ValueError("Unsupported element in generator.")
    rows = [
        [1 if e == x else 0 for e in generator] + [1 if e == z else 0 for e in generator]
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
        if not is_commuting(stab1, stab2):
            not_commuting_pairs.append((stab1, stab2))
    if not_commuting_pairs:
        raise ValueError(f"Following stabilizers do not commute:\n{not_commuting_pairs}")
    if not are_independent(stabilizers):
        raise ValueError("The set of stabilizers are not independent.")


def check_xj(x_j: Operator, z_j: Operator, other_z: list[Operator], stabilizers: list[Operator]) -> None:
    """Check if logical x_j fulfills the commutation relations."""
    non_commuting_stabilizers = []
    for stabilizer in stabilizers:
        if not is_commuting(x_j, stabilizer):
            non_commuting_stabilizers.append(stabilizer)
    if non_commuting_stabilizers:
        raise ValueError(f"x_j does not commute with the following stabilizers:\n{non_commuting_stabilizers}")
    non_commuting_z = []
    for z in other_z:
        if not is_commuting(x_j, z):
            non_commuting_z.append(z)
    if non_commuting_z:
        raise ValueError(f"x_j does not commute with the following z operators:\n{non_commuting_z}")
    if not is_anticommuting(x_j, z_j):
        raise ValueError(f"x_j does not anticommute with z_j")


stabilizers = [
    get_stabilizer(12, x_positions=[1, 4, 5, 7, 8, 10]),
    get_stabilizer(12, x_positions=[2, 3, 6, 9, 11, 12]),
    get_stabilizer(12, x_positions=[2, 5, 6, 8, 9, 11]),
    get_stabilizer(12, x_positions=[3, 4, 6, 7, 9, 12]),
    get_stabilizer(12, z_positions=[1, 4, 5, 7, 8, 10]),
    get_stabilizer(12, z_positions=[2, 3, 6, 9, 11, 12]),
    get_stabilizer(12, z_positions=[2, 5, 6, 8, 9, 11]),
    get_stabilizer(12, z_positions=[3, 4, 6, 7, 9, 12]),
]

print("Checking independency and commuting of stabilizers.")
check_independent_commuting(stabilizers)

z_1 = get_stabilizer(12, z_positions=[4, 7])
z_2 = get_stabilizer(12, z_positions=[5, 8])
z_3 = get_stabilizer(12, z_positions=[1, 5, 9, 12])
z_4 = get_stabilizer(12, z_positions=[2, 6, 7, 10])

print("Checking if logical z operators form with stabilizers an independent and commuting set.")
check_independent_commuting(stabilizers + [z_1, z_2, z_3, z_4])

x_1 = get_stabilizer(12, x_positions=[1, 4, 9, 11])
check_xj(x_1, z_1, [z_2, z_3, z_4], stabilizers)
x_2 = get_stabilizer(12, x_positions=[3, 6, 8, 10])
check_xj(x_2, z_2, [z_1, z_3, z_4], stabilizers)
x_3 = get_stabilizer(12, x_positions=[5, 8])
check_xj(x_3, z_3, [z_1, z_2, z_4], stabilizers)
x_4 = get_stabilizer(12, x_positions=[4, 7])
check_xj(x_4, z_4, [z_1, z_2, z_3], stabilizers)
