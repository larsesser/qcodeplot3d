"""Construct dual graph representations of rectangular and square 2D color codes."""

import dataclasses
import itertools

import rustworkx

from qcodeplot3d.common.construction import PreDualGraphNode, add_edge, coloring_qubits
from qcodeplot3d.common.stabilizers import Color


@dataclasses.dataclass
class TriangularNode2D:
    color: Color
    col: int  # x coordinate
    row: int  # y coordinate
    pre_dg_node: PreDualGraphNode = dataclasses.field(init=False)

    def __post_init__(self):
        self.pre_dg_node = PreDualGraphNode(f"({self.col}, {self.row})", is_boundary=False)

    @property
    def index(self) -> int:
        return self.pre_dg_node.index

    @index.setter
    def index(self, index: int) -> None:
        self.pre_dg_node.index = index

    def __hash__(self):
        return hash((self.col, self.row, self.color))

    @property
    def coordinate(self) -> tuple[int, int]:
        return self.col, self.row


def triangular_node_position(num_rows: int, offset: tuple[int, int] = (0, 0), only: str = None) -> list[tuple[int, int]]:
    """Position of nodes of one color. Top of triangle is (0, 0).

    :param only: only node positions at right, left or center boundary.
    """
    ret: list[tuple[int, int]] = []
    if only is None:
        for row in range(0, 2*num_rows, 2):
            for col in range(-row//2, row//2+1, 2):
                ret.append((col+offset[0], row+offset[1]))
    elif only == "left":
        for row in range(0, 2*num_rows, 2):
            ret.append((-row//2 +offset[0], row+offset[1]))
    elif only == "right":
        for row in range(0, 2*num_rows, 2):
            ret.append((+row//2 +offset[0], row+offset[1]))
    elif only == "center":
        row = 2*num_rows-2
        for col in range(-row // 2, row // 2 + 1, 2):
            ret.append((col + offset[0], row + offset[1]))
    else:
        raise NotImplementedError
    return ret


def triangular_2d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if not distance % 2 == 1:
        raise ValueError("d must be an odd integer")

    num_layer = (distance - 1) // 2

    all_nodes: list[TriangularNode2D] = []
    all_edges: set[tuple[TriangularNode2D, TriangularNode2D]] = set()

    # create boundaries
    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    center = PreDualGraphNode("center", is_boundary=True)
    all_boundary_edges: list[tuple[PreDualGraphNode, PreDualGraphNode]] = []
    for b1, b2 in itertools.combinations([left, right, center], 2):
        all_boundary_edges.append((b1, b2))

    red_offset = (0, 0)
    red_nodes = [TriangularNode2D(Color.red, col, row) for col, row in triangular_node_position(num_layer, red_offset)]
    green_offset = (0, 2)
    green_nodes = [TriangularNode2D(Color.green, col, row) for col, row in triangular_node_position(num_layer, green_offset)]
    yellow_offset = (1, 1)
    yellow_nodes = [TriangularNode2D(Color.yellow, col, row) for col, row in triangular_node_position(num_layer, yellow_offset)]
    all_nodes.extend(red_nodes + green_nodes + yellow_nodes)
    rgy_layer = {node.coordinate: node for node in [*red_nodes, *green_nodes, *yellow_nodes]}

    # add nodes between layer and boundary nodes
    # left
    for offset, layer_dict in [(red_offset, rgy_layer), (green_offset, rgy_layer)]:
        for position in triangular_node_position(num_layer, offset, only="left"):
            all_boundary_edges.append((left, layer_dict[position].pre_dg_node))
    # right
    for offset, layer_dict in [(red_offset, rgy_layer), (yellow_offset, rgy_layer)]:
        for position in triangular_node_position(num_layer, offset, only="right"):
            all_boundary_edges.append((right, layer_dict[position].pre_dg_node))
    # center
    for offset, layer_dict in [(green_offset, rgy_layer), (yellow_offset, rgy_layer)]:
        for position in triangular_node_position(num_layer, offset, only="center"):
            all_boundary_edges.append((center, layer_dict[position].pre_dg_node))

    # add edges between nodes of this layer
    for (x, y), node in rgy_layer.items():
        # add trivial neighbours
        for x_offset in [-1, +1]:
            for y_offset in [-1, 0, +1]:
                if n := rgy_layer.get((x+x_offset, y+y_offset)):
                    all_edges.add((node, n))
        # add neighbour above / below. If there is no node, add next-next neighbour
        if n := rgy_layer.get((x, y+1)):
            all_edges.add((node, n))
        elif n := rgy_layer.get((x, y+2)):
            all_edges.add((node, n))
        if n := rgy_layer.get((x, y-1)):
            all_edges.add((node, n))
        elif n := rgy_layer.get((x, y-2)):
            all_edges.add((node, n))

    graph = rustworkx.PyGraph(multigraph=False)
    for node in [left, right, center]:
        index = graph.add_node(node)
        node.index = index
    for node in all_nodes:
        index = graph.add_node(node.pre_dg_node)
        node.index = index
    for edge in all_boundary_edges:
        add_edge(graph, edge[0], edge[1])
    for edge in all_edges:
        add_edge(graph, edge[0].pre_dg_node, edge[1].pre_dg_node)

    coloring_qubits(graph, dimension=2, do_coloring=True)
    return graph


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

    coloring_qubits(dual_graph, dimension=2)
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

    coloring_qubits(dual_graph, dimension=2)
    return dual_graph
