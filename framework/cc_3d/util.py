import collections
import itertools

import rustworkx
from pysat.formula import CNF
from pysat.solvers import Solver

from framework.cc_3d.base import DualGraphEdge, DualGraphNode
from framework.stabilizers import Color


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


def _compute_coloring(graph: rustworkx.PyGraph, colors: list[Color]) -> dict[int, Color]:
    """Determine a coloring of the given graph with the given colors."""
    cnf = CNF()
    # pysat can not handle expressions containing 0
    offset = 1
    color_offset = len(graph.nodes())
    # each node has exactly one color
    for node in graph.nodes():
        # at least one color
        cnf.append([i * color_offset + node.index + offset for i in range(len(colors))])
        # at most one color
        for a, b in itertools.combinations([i*color_offset+node.index for i in range(len(colors))], 2):
            cnf.append([-(a + offset), -(b + offset)])
    # add a constraint for each edge, so connected nodes do not share the same color
    for _, (node_index1, node_index2, _) in graph.edge_index_map().items():
        for i in range(len(colors)):
            cnf.append([-(i * color_offset + node_index1 + offset), -(i * color_offset + node_index2 + offset)])
    # determine the color of the first volume, to make the solution stable
    cnf.append([min(graph.node_indices())+offset])

    with Solver(bootstrap_with=cnf) as solver:
        solver.solve()
        solution = solver.get_model()

    coloring: dict[int, Color] = {}
    for var in solution:
        if var < 0:
            continue
        for i in reversed(range(len(colors))):
            if (index := var - (i * color_offset + offset)) >= 0:
                coloring[index] = colors[i]
                break
    return coloring


def coloring_qubits(dual_graph: rustworkx.PyGraph, dimension: int = 3, do_coloring: bool = True) -> None:
    # In the following, we construct all necessary information from the graph layout:
    # - color of the nodes
    # - adjacent qubits to stabilizers / boundaries

    if dimension not in {2, 3}:
        raise NotImplementedError

    boundary_nodes_indices = {node.index for node in dual_graph.nodes() if node.is_boundary}

    # compute a coloring of the graph
    colors = [Color.red, Color.blue, Color.green]
    if dimension == 3:
        colors.append(Color.yellow)
    if do_coloring:
        coloring = _compute_coloring(dual_graph, colors)

    # find all triangles / tetrahedrons of the graph
    simplexes = compute_simplexes(dual_graph, dimension, exclude_boundary_simplexes=True)
    simplex_map = {simplex: number for number, simplex in enumerate(simplexes, start=1)}
    all_qubits = sorted(simplex_map.values())
    node2simplex = collections.defaultdict(list)
    for simplex, name in simplex_map.items():
        for index in simplex:
            node2simplex[index].append(name)

    max_id = 0

    # add proper DualGraphNode objects for all graph nodes
    for node in dual_graph.nodes():
        if do_coloring:
            color = coloring[node.index]
        else:
            color = Color.green
        qubits = node2simplex[node.index]
        dual_graph[node.index] = DualGraphNode(max_id, color, qubits, is_stabilizer=(node.index not in boundary_nodes_indices), all_qubits=all_qubits)
        max_id += 1
        dual_graph[node.index].index = node.index
        dual_graph[node.index].title = node.title

    # add proper DualGraphEdge objects for all graph edges
    for edge_index, (node_index1, node_index2, _) in dual_graph.edge_index_map().items():
        edge = DualGraphEdge(max_id, dual_graph[node_index1], dual_graph[node_index2])
        max_id += 1
        edge.index = edge_index
        dual_graph.update_edge_by_index(edge_index, edge)
