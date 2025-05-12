"""Generic helpers to construct color codes. Used in cc_2d/construction.py and cc_3d/construction.py."""

import collections
import itertools
from typing import Optional

import rustworkx
from pysat.formula import CNF
from pysat.solvers import Solver

from framework.common import compute_simplexes
from framework.common.graph import (
    DualGraphEdge,
    DualGraphNode,
    GraphNode,
    RestrictedGraphEdge,
    RestrictedGraphNode,
)
from framework.common.stabilizers import Color


class PreDualGraphNode(GraphNode):
    def __init__(self, title: str, is_boundary: bool = False):
        self._is_boundary = is_boundary
        self.title = title

    @property
    def is_boundary(self) -> bool:
        return self._is_boundary

    def __repr__(self):
        return self.title


def add_edge(graph: rustworkx.PyGraph, node1: Optional[PreDualGraphNode], node2: Optional[PreDualGraphNode]) -> None:
    """Helper function to add an edge only if both nodes are not None."""
    if node1 is None or node2 is None:
        return
    graph.add_edge(node1.index, node2.index, None)


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
    # determine the color of two boundaries, to make the solution stable
    boundary_nodes = [node for node in graph.nodes() if node.is_boundary]
    if len(boundary_nodes) <= 4:
        cnf.append([boundary_nodes[0].index+offset])
    elif len(boundary_nodes) == 6:
        cnf.append([boundary_nodes[0].index+offset])
        neighboured_boundaries = [node for node in boundary_nodes if graph.has_edge(boundary_nodes[0].index, node.index)]
        cnf.append([neighboured_boundaries[0].index+color_offset+offset])
    else:
        raise NotImplementedError

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
        dual_graph[node.index] = DualGraphNode(
            max_id, color, qubits, is_stabilizer=(node.index not in boundary_nodes_indices), all_qubits=all_qubits)
        max_id += 1
        dual_graph[node.index].index = node.index
        dual_graph[node.index].title = node.title

    # add proper DualGraphEdge objects for all graph edges
    for edge_index, (node_index1, node_index2, _) in dual_graph.edge_index_map().items():
        edge = DualGraphEdge(max_id, dual_graph[node_index1], dual_graph[node_index2])
        max_id += 1
        edge.index = edge_index
        dual_graph.update_edge_by_index(edge_index, edge)


def construct_restricted_graph(dual_graph: rustworkx.PyGraph, colors: list[Color]) -> rustworkx.PyGraph:
    graph = rustworkx.PyGraph(multigraph=False)
    for node in dual_graph.nodes():
        if node.color in colors:
            graph.add_node(RestrictedGraphNode(node))
    for index in graph.node_indices():
        graph[index].index = index
    nodes_by_id = {node.id: node for node in graph.nodes()}
    # take care to get the properties of the nodes associated to an edge right
    for edge in dual_graph.edges():
        if edge.node1.color in colors and edge.node2.color in colors:
            rg_edge = RestrictedGraphEdge(nodes_by_id[edge.node1.id], nodes_by_id[edge.node2.id], edge)
            graph.add_edge(rg_edge.node1.index, rg_edge.node2.index, rg_edge)
    for index in graph.edge_indices():
        graph.edge_index_map()[index][2].index = index
    return graph
