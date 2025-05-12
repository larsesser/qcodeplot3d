"""Provide some utility function and classes which are used in multiple files."""

import enum
import itertools

import rustworkx


class Kind(enum.Enum):
    """Which kind of stabilizer?"""
    x = "x"
    z = "z"

    @property
    def opposite(self) -> "Kind":
        if self == Kind.x:
            return Kind.z
        if self == Kind.z:
            return Kind.x
        raise NotImplementedError

    def __str__(self):
        return self.value


def compute_simplexes(graph: rustworkx.PyGraph, dimension: int, exclude_boundary_simplexes: bool = False) -> set[tuple[int, ...]]:
    """Find all simplexes of the given dimension in the graph.

    param exclude_boundary_simplexes: If True, exclude simplexes which vertices are all boundary vertices.
    """
    if dimension not in {2, 3}:
        raise NotImplementedError
    triangles = set()
    filtered_triangles = set()
    for node1 in graph.nodes():
        node1_neighbors = graph.neighbors(node1.index)
        for node2_index in node1_neighbors:
            for node3_index in graph.neighbors(node2_index):
                if node3_index not in node1_neighbors:
                    continue
                triangle = tuple(sorted([node1.index, node2_index, node3_index]))
                triangles.add(triangle)
                # exclude triangles between only boundary nodes
                if exclude_boundary_simplexes and all(graph.nodes()[index].is_boundary for index in triangle):
                    continue
                filtered_triangles.add(triangle)
    if dimension == 2:
        return filtered_triangles
    tetrahedrons = set()
    for triangle in triangles:
        common_neighbours = set(graph.neighbors(triangle[0])) & set(graph.neighbors(triangle[1])) & set(graph.neighbors(triangle[2]))
        for neighbour in common_neighbours:
            tetrahedron = tuple(sorted([*triangle, neighbour]))
            # exclude tetrahedrons between only boundary nodes
            if exclude_boundary_simplexes and all(graph.nodes()[index].is_boundary for index in tetrahedron):
                continue
            tetrahedrons.add(tetrahedron)
    return tetrahedrons


def powerset(iterable):
    """powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
