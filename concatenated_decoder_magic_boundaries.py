import collections

import rustworkx

from framework.decoder import ConcatenatedDecoder, GraphNode, GraphEdge, DualGraphNode, DualGraphEdge
from rustworkx.visualization import graphviz_draw
from framework.stabilizers import Color, Stabilizer
import rustworkx as rx
import itertools
from pysat.formula import CNF
from pysat.solvers import Solver


class PreDualGraphNode(GraphNode):
    title: str
    # used for graph rendering
    initial_position: tuple[int, int] = None

    def __init__(self, title: str, pos: tuple[int, int] = None, is_boundary: bool = False):
        self._is_boundary = is_boundary
        self.title = title
        self.initial_position = pos

    @property
    def is_boundary(self) -> bool:
        return self._is_boundary


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
        DualGraphNode(Color.red, Stabilizer(15, Color.red, x_positions=t2p(["A", "J", "K", "H", "M", "Q", "I", "P"]))),
        DualGraphNode(Color.yellow, Stabilizer(15, Color.yellow, x_positions=t2p(["B", "F", "N", "G", "J", "K", "M", "Q"]))),
        DualGraphNode(Color.blue, Stabilizer(15, Color.blue, x_positions=t2p(["F", "C", "E", "N", "M", "I", "P", "Q"]))),
        DualGraphNode(Color.green, Stabilizer(15, Color.green, x_positions=t2p(["D", "G", "N", "E", "H", "K", "Q", "P"]))),
        DualGraphNode(Color.red, None, t2p(["B", "F", "C", "E", "D", "G", "N"])),
        DualGraphNode(Color.yellow, None, t2p(["A", "I", "C", "E", "D", "H", "P"])),
        DualGraphNode(Color.blue, None, t2p(["A", "J", "B", "G", "D", "H", "K"])),
        DualGraphNode(Color.green, None, t2p(["A", "J", "B", "F", "C", "I", "M"])),
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
    boundary_nodes_indices = [nodes[4].index, nodes[5].index, nodes[6].index, nodes[7].index]

    coloring_qubits(dual_graph, boundary_nodes_indices, dimension=3)
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


def _compute_simplexes(graph: rustworkx.PyGraph, dimension: int) -> set[tuple[int, ...]]:
    if dimension not in {2, 3}:
        raise NotImplementedError
    triangles = set()
    for node1 in graph.nodes():
        node1_neighbors = graph.neighbors(node1.index)
        for node2_index in node1_neighbors:
            for node3_index in graph.neighbors(node2_index):
                if node3_index not in node1_neighbors:
                    continue
                triangles.add(tuple(sorted([node1.index, node2_index, node3_index])))
    if dimension == 2:
        return triangles
    tetrahedrons = set()
    for triangle in triangles:
        common_neighbours = set(graph.neighbors(triangle[0])) & set(graph.neighbors(triangle[1])) & set(graph.neighbors(triangle[2]))
        for neighbour in common_neighbours:
            tetrahedrons.add(tuple(sorted([*triangle, neighbour])))
    return tetrahedrons


def coloring_qubits(dual_graph: rustworkx.PyGraph, boundary_nodes_indices: list[int] = None, dimension: int = 3) -> None:
    # In the following, we construct all necessary information from the graph layout:
    # - color of the nodes
    # - adjacent qubits to stabilizers / boundaries

    if dimension not in {2, 3}:
        raise NotImplementedError

    if boundary_nodes_indices is None:
        boundary_nodes_indices = {node.index for node in dual_graph.nodes() if node.is_boundary}

    # rustworkx guarantees that the undirected and directed graph share the same indexes for the same objects.
    directed_dual_graph = dual_graph.to_directed()

    # compute a coloring of the graph
    colors = [Color.red, Color.blue, Color.green]
    if dimension == 3:
        colors.append(Color.yellow)
    coloring = _compute_coloring(dual_graph, colors)

    # find all triangles / tetrahedrons of the graph
    simplexes = _compute_simplexes(dual_graph, dimension)
    simplex_map = {simplex: number for number, simplex in enumerate(simplexes, start=1)}
    node2simplex = collections.defaultdict(list)
    for simplex, name in simplex_map.items():
        for index in simplex:
            node2simplex[index].append(name)

    # add proper DualGraphNode objects for all graph nodes
    for index in dual_graph.node_indices():
        color = coloring[index]
        adjacent_qubits = node2tetrahedron[index]
        if index in boundary_nodes_indices:
            dual_graph[index] = DualGraphNode(color, stabilizer=None, _adjacent_qubits=adjacent_qubits)
        else:
            stabilizer = Stabilizer(length=len(tetrahedrons), color=color, x_positions=adjacent_qubits)
            dual_graph[index] = DualGraphNode(color, stabilizer=stabilizer)
        dual_graph[index].index = index

    # add proper DualGraphEdge objects for all graph edges
    for edge_index, (node_index1, node_index2, _) in dual_graph.edge_index_map().items():
        edge = DualGraphEdge(dual_graph[node_index1], dual_graph[node_index2])
        edge.index = edge_index
        dual_graph.update_edge_by_index(edge_index, edge)


def rectangular_2d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if not distance % 2 == 1:
        raise ValueError("d must be an odd integer")

    num_cols = distance-1
    num_rows = distance

    dual_graph = rustworkx.PyGraph(multigraph=False)
    left = PreDualGraphNode("left", pos=(-distance//2, distance//2), is_boundary=True)
    right = PreDualGraphNode("right", pos=(3*distance//2, distance//2), is_boundary=True)
    top = PreDualGraphNode("top", pos=(distance//2, 3*distance//2), is_boundary=True)
    bottom = PreDualGraphNode("bottom", pos=(distance//2, -distance//2), is_boundary=True)
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
    nodes = [[PreDualGraphNode(f"({col},{row})", pos=(col, row)) for row in reversed(range(num_rows))] for col in range(num_cols)]
    for row in range(1, num_rows, 2):
        nodes[0][row] = None

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for node in itertools.chain.from_iterable(nodes) if node is not None])
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # construct edges

    # between boundary_nodes and boundary_nodes:
    dual_graph.add_edge(left.index, top.index, None)
    dual_graph.add_edge(top.index, right.index, None)
    dual_graph.add_edge(right.index, bottom.index, None)
    dual_graph.add_edge(bottom.index, left.index, None)

    # between nodes and boundary_nodes
    for col in nodes:
        dual_graph.add_edge(col[0].index, top.index, None)
        dual_graph.add_edge(col[-1].index, bottom.index, None)
    for row in range(num_rows):
        if row % 2 == 0:
            dual_graph.add_edge(nodes[0][row].index, left.index, None)
        else:
            dual_graph.add_edge(nodes[1][row].index, left.index, None)
    for node in nodes[-1]:
        dual_graph.add_edge(node.index, right.index, None)

    # between nodes and nodes
    # connect rows
    for col1, col2 in zip(nodes, nodes[1:]):
        for node1, node2 in zip(col1, col2):
            if node1 is None or node2 is None:
                continue
            dual_graph.add_edge(node1.index, node2.index, None)
    # connect cols
    for col in nodes:
        for node1, node2 in zip(col, col[1:]):
            if node1 is None or node2 is None:
                continue
            dual_graph.add_edge(node1.index, node2.index, None)

    for col_pos, col in enumerate(nodes):
        # reached last col
        if col_pos == num_cols-1:
            continue
        for row_pos, node in enumerate(col):
            # diagonal pattern, including all odd rows from even cols and vice versa
            if row_pos % 2 != col_pos % 2:
                continue
            if row_pos != num_rows-1:
                dual_graph.add_edge(node.index, nodes[col_pos+1][row_pos+1].index, None)
            if row_pos != 0:
                dual_graph.add_edge(node.index, nodes[col_pos+1][row_pos-1].index, None)
    return dual_graph


def node_attr_fn(node: GraphNode):
    label = f"{node.index}"
    if hasattr(node, "title"):
        label = f"{node.title}"
    if node.is_boundary:
        label += " B"
    attr_dict = {
        "style": "filled",
        "shape": "circle",
        "label": label,
    }
    if hasattr(node, "color"):
        attr_dict["color"] = node.color.name
        attr_dict["fill_color"] = node.color.name
    if hasattr(node, "initial_position"):
        x, y = node.initial_position
        attr_dict["pos"] = f'"{x},{y}"'
    return attr_dict


def edge_attr_fn(edge: GraphEdge):
    attr_dict = {
        "label": f"{edge.id}, {edge.index}",
    }
    return attr_dict


graph = rectangular_2d_dual_graph(5)
coloring_qubits(graph, dimension=2)
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
