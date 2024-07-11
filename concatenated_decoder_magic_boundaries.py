import collections

import numpy as np
import rustworkx
import shlex
import re
from typing import Optional

from framework.stabilizers import check_stabilizers, Operator, check_z, check_xj, count_independent
from framework.decoder import ConcatenatedDecoder, GraphNode, GraphEdge, DualGraphNode, DualGraphEdge
from rustworkx.visualization import graphviz_draw
from framework.stabilizers import Color, Stabilizer
import rustworkx as rx
import itertools
from pysat.formula import CNF
from pysat.solvers import Solver
from framework.plotter import Plotter3D, compute_simplexes


class PreDualGraphNode(GraphNode):
    def __init__(self, title: str, is_boundary: bool = False):
        self._is_boundary = is_boundary
        self.title = title

    @property
    def is_boundary(self) -> bool:
        return self._is_boundary

    def __repr__(self):
        return self.title


def construct_dual_graph() -> rustworkx.PyGraph:
    # dual_graph = rustworkx.PyGraph(multigraph=False)
    # boundary_nodes = [PreDualGraphNode(is_boundary=True) for _ in range(6)]
    # boundary_nodes_indices = dual_graph.add_nodes_from(boundary_nodes)
    # central_node = PreDualGraphNode()
    # TODO finish construction of concrete dual graph

    # hard coded construction of [[15,1,3]] 3D Tetrahedral Color Code
    def t2p(l_in: list[str]) -> list[int]:
        trans = {"A": 15, "B": 10, "C": 4, "D": 6, "E": 5, "F": 8, "G": 9, "H": 3, "I": 1, "J": 7, "K": 12, "M": 14,
                 "N": 11, "P": 2, "Q": 13}
        return [trans[e] for e in l_in]

    dual_graph = rustworkx.PyGraph(multigraph=False)
    nodes = [
        DualGraphNode(Color.red, t2p(["A", "J", "K", "H", "M", "Q", "I", "P"]), is_stabilizer=True, stabilizer_length=15),
        DualGraphNode(Color.yellow, t2p(["B", "F", "N", "G", "J", "K", "M", "Q"]), is_stabilizer=True, stabilizer_length=15),
        DualGraphNode(Color.blue, t2p(["F", "C", "E", "N", "M", "I", "P", "Q"]), is_stabilizer=True, stabilizer_length=15),
        DualGraphNode(Color.green, t2p(["D", "G", "N", "E", "H", "K", "Q", "P"]), is_stabilizer=True, stabilizer_length=15),
        DualGraphNode(Color.red, t2p(["B", "F", "C", "E", "D", "G", "N"]), is_stabilizer=False),
        DualGraphNode(Color.yellow, t2p(["A", "I", "C", "E", "D", "H", "P"]), is_stabilizer=False),
        DualGraphNode(Color.blue, t2p(["A", "J", "B", "G", "D", "H", "K"]), is_stabilizer=False),
        DualGraphNode(Color.green, t2p(["A", "J", "B", "F", "C", "I", "M"]), is_stabilizer=False),
    ]
    dual_graph.add_nodes_from(nodes)
    for index in dual_graph.node_indices():
        dual_graph[index].index = index
    # insert edges between the nodes
    for node1, node2 in itertools.combinations(dual_graph.nodes(), 2):
        if node1.color == node2.color or dual_graph.has_edge(node1.index, node2.index):
            continue
        dual_graph.add_edge(node1.index, node2.index, DualGraphEdge(node1, node2))
    for index in dual_graph.edge_indices():
        dual_graph.edge_index_map()[index][2].index = index

    coloring_qubits(dual_graph, dimension=3)
    return dual_graph


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


def coloring_qubits(dual_graph: rustworkx.PyGraph, dimension: int = 3) -> None:
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
    coloring = _compute_coloring(dual_graph, colors)

    # find all triangles / tetrahedrons of the graph
    simplexes = compute_simplexes(dual_graph, dimension)
    simplex_map = {simplex: number for number, simplex in enumerate(simplexes, start=1)}
    node2simplex = collections.defaultdict(list)
    for simplex, name in simplex_map.items():
        for index in simplex:
            node2simplex[index].append(name)

    # add proper DualGraphNode objects for all graph nodes
    for node in dual_graph.nodes():
        color = coloring[node.index]
        qubits = node2simplex[node.index]
        if node.index in boundary_nodes_indices:
            dual_graph[node.index] = DualGraphNode(color, qubits, is_stabilizer=False)
        else:
            dual_graph[node.index] = DualGraphNode(color, qubits, is_stabilizer=True, stabilizer_length=len(simplexes))
        dual_graph[node.index].index = node.index
        dual_graph[node.index].title = node.title

    # add proper DualGraphEdge objects for all graph edges
    for edge_index, (node_index1, node_index2, _) in dual_graph.edge_index_map().items():
        edge = DualGraphEdge(dual_graph[node_index1], dual_graph[node_index2])
        edge.index = edge_index
        dual_graph.update_edge_by_index(edge_index, edge)


def add_edge(graph: rustworkx.PyGraph, node1: Optional[PreDualGraphNode], node2: Optional[PreDualGraphNode]) -> None:
    """Helper function to add an edge only if both nodes are not None."""
    if node1 is None or node2 is None:
        return
    graph.add_edge(node1.index, node2.index, None)


def rectangular_2d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if not distance % 2 == 1:
        raise ValueError("d must be an odd integer")

    num_cols = distance-1
    num_rows = distance

    dual_graph = rustworkx.PyGraph(multigraph=False)
    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    top = PreDualGraphNode("top", is_boundary=True)
    bottom = PreDualGraphNode("bottom", is_boundary=True)
    boundaries = [left, right, top, bottom]
    # distance 3:
    #   | |
    # – a b –
    #    \|
    # ––– c –
    #    /|
    # – d e –
    #   | |
    # nodes = [[a, None, d], [b, c, e]]
    nodes = [[PreDualGraphNode(f"({col},{row})") for row in reversed(range(num_rows))] for col in range(num_cols)]
    for row in range(1, num_rows, 2):
        nodes[0][row] = None

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for node in itertools.chain.from_iterable(nodes) if node is not None])
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # construct edges

    # between boundary_nodes and boundary_nodes:
    add_edge(dual_graph, left, top)
    add_edge(dual_graph, top, right)
    add_edge(dual_graph, right, bottom)
    add_edge(dual_graph, bottom, left)

    # between nodes and boundary_nodes
    for col in nodes:
        add_edge(dual_graph, col[0], top)
        add_edge(dual_graph, col[-1], bottom)
    for row in range(num_rows):
        if row % 2 == 0:
            add_edge(dual_graph, nodes[0][row], left)
        else:
            add_edge(dual_graph, nodes[1][row], left)
    for node in nodes[-1]:
        add_edge(dual_graph, node, right)

    # between nodes and nodes
    # connect rows
    for col1, col2 in zip(nodes, nodes[1:]):
        for node1, node2 in zip(col1, col2):
            add_edge(dual_graph, node1, node2)
    # connect cols
    for col in nodes:
        for node1, node2 in zip(col, col[1:]):
            add_edge(dual_graph, node1, node2)

    for col_pos, col in enumerate(nodes):
        # reached last col
        if col_pos == num_cols-1:
            continue
        for row_pos, node in enumerate(col):
            # diagonal pattern, including all odd rows from even cols and vice versa
            if row_pos % 2 != col_pos % 2:
                continue
            if row_pos != num_rows-1:
                add_edge(dual_graph, node, nodes[col_pos+1][row_pos+1])
            if row_pos != 0:
                add_edge(dual_graph, node, nodes[col_pos+1][row_pos-1])
    return dual_graph


def square_2d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if not distance % 2 == 0:
        raise ValueError("d must be an even integer")

    num_cols = num_rows = distance-1

    dual_graph = rustworkx.PyGraph(multigraph=False)
    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    top = PreDualGraphNode("top", is_boundary=True)
    bottom = PreDualGraphNode("bottom", is_boundary=True)
    boundaries = [left, right, top, bottom]
    # distance 4:
    #   |   |   |
    # – a - b - c –
    #   | / | \ |
    # – d – e - f -
    #   | \ | / |
    # – g - h - i -
    #   |   |   |
    # nodes = [[a, d, g], [b, e, h], [c, f, i]]
    nodes = [[PreDualGraphNode(f"({col},{row})") for row in reversed(range(num_rows))] for col in range(num_cols)]

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for node in itertools.chain.from_iterable(nodes) if node is not None])
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # construct edges

    # between boundary_nodes and boundary_nodes:
    add_edge(dual_graph, left, top)
    add_edge(dual_graph, top, right)
    add_edge(dual_graph, right, bottom)
    add_edge(dual_graph, bottom, left)

    # between nodes and boundary_nodes
    for col in nodes:
        add_edge(dual_graph, col[0], top)
        add_edge(dual_graph, col[-1], bottom)
    for node in nodes[0]:
        add_edge(dual_graph, node, left)
    for node in nodes[-1]:
        add_edge(dual_graph, node, right)

    # between nodes and nodes
    # connect rows
    for col1, col2 in zip(nodes, nodes[1:]):
        for node1, node2 in zip(col1, col2):
            add_edge(dual_graph, node1, node2)
    # connect cols
    for col in nodes:
        for node1, node2 in zip(col, col[1:]):
            add_edge(dual_graph, node1, node2)

    for col_pos, col in enumerate(nodes):
        # reached last col
        if col_pos == num_cols-1:
            continue
        for row_pos, node in enumerate(col):
            # diagonal pattern, including all odd rows from even cols and vice versa
            if row_pos % 2 == col_pos % 2:
                continue
            if row_pos != num_rows-1:
                add_edge(dual_graph, node, nodes[col_pos+1][row_pos+1])
            if row_pos != 0:
                add_edge(dual_graph, node, nodes[col_pos+1][row_pos-1])
    return dual_graph


def cubic_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if not distance % 2 == 0:
        raise ValueError("d must be an even integer")

    num_cols = num_rows = num_layers = distance-1

    dual_graph = rustworkx.PyGraph(multigraph=False)
    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    back = PreDualGraphNode("back", is_boundary=True)
    front = PreDualGraphNode("front", is_boundary=True)
    top = PreDualGraphNode("top", is_boundary=True)
    bottom = PreDualGraphNode("bottom", is_boundary=True)
    boundaries = [left, right, back, front, top,  bottom]
    # distance 4, top layer:
    #   |   |   |
    # – a - b - c –
    #   | / | \ |
    # – d – e - f -
    #   | \ | / |
    # – g - h - i -
    #   |   |   |
    # distance 4, middle layer:
    #   |   |   |
    # – j - k - l –
    #   | \ | / |
    # – m – n - o -
    #   | / | \ |
    # – p - q - r -
    #   |   |   |
    # distance 4, bottom layer:
    #   |   |   |
    # – r - t - u –
    #   | / | \ |
    # – v – w - x -
    #   | \ | / |
    # – y - z - A -
    #   |   |   |
    # nodes = [[[a, d, g], [b, e, h], [c, f, i]], [[j, m, p], [k, n, q], [l, o, r]], [[r, v, y], [t, w, z], [u, x, A]]]
    nodes = [[[PreDualGraphNode(f"({col},{row},{layer})")
               for row in range(num_rows)] for col in range(num_cols)] for layer in range(num_layers)]

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for layer in nodes for row in layer for node in row])
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # construct edges

    # between boundary_nodes and boundary_nodes:
    add_edge(dual_graph, left, back)
    add_edge(dual_graph, back, right)
    add_edge(dual_graph, right, front)
    add_edge(dual_graph, front, left)
    for node in [left, right, back, front]:
        add_edge(dual_graph, top, node)
        add_edge(dual_graph, bottom, node)

    # between nodes and boundary_nodes
    for layer in nodes:
        for col in layer:
            add_edge(dual_graph, col[0], front)
            add_edge(dual_graph, col[-1], back)
        # first row
        for node in layer[0]:
            add_edge(dual_graph, node, left)
        # last row
        for node in layer[-1]:
            add_edge(dual_graph, node, right)
    for col in nodes[0]:
        for node in col:
            add_edge(dual_graph, node, top)
    for col in nodes[-1]:
        for node in col:
            add_edge(dual_graph, node, bottom)

    # between nodes and nodes
    # inside one layer
    for layer_pos, layer in enumerate(nodes):
        # connect rows
        for col1, col2 in zip(layer, layer[1:]):
            for node1, node2 in zip(col1, col2):
                add_edge(dual_graph, node1, node2)
        # connect cols
        for col in layer:
            for node1, node2 in zip(col, col[1:]):
                add_edge(dual_graph, node1, node2)
        # diagonals
        for col_pos, col in enumerate(layer):
            # reached last col
            if col_pos == num_cols-1:
                continue
            for row_pos, node in enumerate(col):
                # diagonal pattern, changing "direction" in each layer
                if (layer_pos % 2 == 0) and (row_pos % 2 == col_pos % 2):
                    continue
                if (layer_pos % 2 == 1) and (row_pos % 2 != col_pos % 2):
                    continue
                if row_pos != num_rows-1:
                    add_edge(dual_graph, node, layer[col_pos+1][row_pos+1])
                if row_pos != 0:
                    add_edge(dual_graph, node, layer[col_pos+1][row_pos-1])
    # between two layers
    for layer1, layer2 in zip(nodes, nodes[1:]):
        for col1, col2 in zip(layer1, layer2):
            for node1, node2 in zip(col1, col2):
                add_edge(dual_graph, node1, node2)
    for layer_pos, layer in enumerate(nodes):
        # reached last layer
        if layer_pos == num_layers-1:
            continue
        for col_pos, col in enumerate(layer):
            for row_pos, node in enumerate(col):
                # diagonal pattern, changing "direction" in each layer
                if (layer_pos % 2 == 0) and (row_pos % 2 == col_pos % 2):
                    continue
                if (layer_pos % 2 == 1) and (row_pos % 2 != col_pos % 2):
                    continue
                if row_pos != num_rows-1:
                    add_edge(dual_graph, node, nodes[layer_pos+1][col_pos][row_pos+1])
                if row_pos != 0:
                    add_edge(dual_graph, node, nodes[layer_pos+1][col_pos][row_pos-1])
                if col_pos != num_cols-1:
                    add_edge(dual_graph, node, nodes[layer_pos+1][col_pos+1][row_pos])
                if col_pos != 0:
                    add_edge(dual_graph, node, nodes[layer_pos+1][col_pos-1][row_pos])

    return dual_graph


def node_attr_fn(node: GraphNode):
    label = f"{node.index}"
    if node.title:
        label = f"{node.title}"
    if node.is_boundary:
        label += " B"
    attr_dict = {
        "style": "filled",
        "shape": "point",
        "label": label,
    }
    if hasattr(node, "color"):
        attr_dict["color"] = node.color.name
        attr_dict["fill_color"] = node.color.name
    return attr_dict


def edge_attr_fn(edge: GraphEdge):
    attr_dict = {
    }
    return attr_dict


graph = cubic_3d_dual_graph(4)
coloring_qubits(graph, dimension=3)

x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]
z_stabilizer: list[Stabilizer] = [edge.stabilizer for edge in graph.edges() if edge.is_stabilizer]
stabilizers: list[Stabilizer] = [*x_stabilizer, *z_stabilizer]

num_independent = count_independent(stabilizers)
print(f"Stabilizers: {len(stabilizers)}, independent: {num_independent}")
odd_stabilizers = [stabilizer for stabilizer in stabilizers if len(stabilizer.qubits) % 2 == 1]
print(f"Odd stabilizers: {len(odd_stabilizers)}")
n = stabilizers[0].length
k = n - num_independent
print(f"n: {n}, k: {k}, expected k: 3")

plotter = Plotter3D(graph, "3D cubic")
plotter.show_debug_dual_mesh(show_labels=True, explode_factor=0.0, exclude_boundaries=True)
plotter.show_primay_mesh(show_labels=True, explode_factor=0.4)

exit()

d = 4
graph = square_2d_dual_graph(d)
coloring_qubits(graph, dimension=2)

x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.stabilizer]
z_stabilizer: list[Stabilizer] = [Stabilizer(s.length, s.color, z_positions=s.x) for s in x_stabilizer]
stabilizers: list[Stabilizer] = [*x_stabilizer, *z_stabilizer]
check_stabilizers(stabilizers)

logical_1_pos = [node for node in graph.nodes() if node.title == "bottom"][0].qubits
z_1 = Operator(length=stabilizers[0].length, z_positions=logical_1_pos)
x_2 = Operator(length=stabilizers[0].length, x_positions=logical_1_pos)

logical_2_pos = [node for node in graph.nodes() if node.title == "left"][0].qubits
z_2 = Operator(length=stabilizers[0].length, z_positions=logical_2_pos)
x_1 = Operator(length=stabilizers[0].length, x_positions=logical_2_pos)

check_z([z_1, z_2], stabilizers)
check_xj(x_1, z_1, [z_2], stabilizers)
check_xj(x_2, z_2, [z_1], stabilizers)

n = stabilizers[0].length
k = n - len(stabilizers)
print(f"n: {n}, k: {k}, d: {d}")

graphviz_draw(graph, node_attr_fn, filename="2D square d=4.png", method="fdp")


exit()

graph = rectangular_2d_dual_graph(5)
coloring_qubits(graph, dimension=2)

x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.stabilizer]
z_stabilizer: list[Stabilizer] = [Stabilizer(s.length, s.color, z_positions=s.x) for s in x_stabilizer]
stabilizers = [*x_stabilizer, *z_stabilizer]
check_stabilizers(stabilizers)

logical_1_pos = [node for node in graph.nodes() if node.title == "bottom"][0].qubits
z_1 = Operator(length=stabilizers[0].length, z_positions=logical_1_pos)
x_1 = Operator(length=stabilizers[0].length, x_positions=logical_1_pos)

logical_2_pos = [node for node in graph.nodes() if node.title == "top"][0].qubits
z_2 = Operator(length=stabilizers[0].length, z_positions=logical_2_pos)
x_2 = Operator(length=stabilizers[0].length, x_positions=logical_2_pos)

check_z([z_1, z_2], stabilizers)
check_xj(x_1, z_1, [z_2], stabilizers)
check_xj(x_2, z_2, [z_1], stabilizers)

graphviz_draw(graph, node_attr_fn, filename="2D rectangular d=3.png", method="fdp")


exit()

decoder = ConcatenatedDecoder([Color.red, Color.green, Color.yellow, Color.blue], construct_dual_graph())

graphviz_draw(decoder.dual_graph, node_attr_fn, edge_attr_fn, filename="mb_dualgraph.png", method="sfdp")
# https://stackoverflow.com/questions/14662618/is-there-a-3d-version-of-graphviz
exit()
# graphviz_draw(decoder.dual_graph, node_attr_fn, edge_attr_fn, filename="mb_dualgraph.vrml", method="sfdp", image_type="vrml", graph_attr={"dimen": "3"})
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow]), node_attr_fn, edge_attr_fn, filename="mb_restricted_gy.png", method="sfdp")
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow, Color.blue]), node_attr_fn, edge_attr_fn, filename="mb_restricted_gyb.png", method="sfdp")
graphviz_draw(decoder.mc3_graph([Color.green, Color.yellow], Color.blue), node_attr_fn, edge_attr_fn, filename="mb_monochrom_gy_b.png", method="sfdp")
graphviz_draw(decoder.mc4_graph([Color.green, Color.yellow], Color.blue, Color.red), node_attr_fn, edge_attr_fn, filename="mb_monochrom_gy_b_r.png", method="sfdp")
