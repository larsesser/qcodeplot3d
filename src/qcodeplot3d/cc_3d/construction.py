"""Construct dual graph representations of tetrahedral and cubic 3D color codes."""

import dataclasses
import itertools

import rustworkx

from qcodeplot3d.cc_2d.construction import TriangularNode2D, triangular_node_position
from qcodeplot3d.common.construction import PreDualGraphNode, add_edge, coloring_qubits
from qcodeplot3d.common.graph import DualGraphNode, XDualGraphEdge, XDualGraphNode
from qcodeplot3d.common.stabilizers import Color, Operator


@dataclasses.dataclass
class TetrahedralNode3D(TriangularNode2D):
    layer: int  # z coordinate

    def __post_init__(self):
        self.pre_dg_node = PreDualGraphNode(f"({self.col}, {self.row}, {self.layer})", is_boundary=False)

    def __hash__(self):
        return hash((self.col, self.row, self.layer, self.color))

    @property
    def coordinate(self) -> tuple[int, int, int]:
        return self.col, self.row, self.layer

    @property
    def coordinate_2d(self) -> tuple[int, int]:
        return self.col, self.row


def tetrahedron_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if not distance % 2 == 1:
        raise ValueError("d must be an odd integer")

    # construct all layers, which are 2D triangular codes of distance d, d-2, d-4 ... to d=3
    num_layers = (distance - 1) // 2

    all_nodes: list[TetrahedralNode3D] = []
    all_edges: set[tuple[TetrahedralNode3D, TetrahedralNode3D]] = set()
    rgy_layers: dict[int, dict[tuple[int, int], TetrahedralNode3D]] = {}
    rgb_layers: dict[int, dict[tuple[int, int], TetrahedralNode3D]] = {}

    # create boundaries
    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    center = PreDualGraphNode("center", is_boundary=True)
    bottom = PreDualGraphNode("bottom", is_boundary=True)
    all_boundary_edges: list[tuple[PreDualGraphNode, PreDualGraphNode]] = []
    for b1, b2 in itertools.combinations([left, right, center, bottom], 2):
        all_boundary_edges.append((b1, b2))

    # create each 2D layer, enumerated from top to bottom, create from bottom to top
    for layer in range(num_layers, 0, -1):
        # create nodes of this layer
        red_offset = (0, 0)
        red_nodes = [TetrahedralNode3D(Color.red, col, row, layer) for col, row in triangular_node_position(layer, red_offset)]
        green_offset = (0, 2)
        green_nodes = [TetrahedralNode3D(Color.green, col, row, layer) for col, row in triangular_node_position(layer, green_offset)]
        yellow_offset = (1, 1)
        yellow_nodes = [TetrahedralNode3D(Color.yellow, col, row, layer) for col, row in triangular_node_position(layer, yellow_offset)]
        blue_nodes = [TetrahedralNode3D(Color.blue, col, row, layer) for col, row in triangular_node_position(layer, yellow_offset)]
        all_nodes.extend(red_nodes + green_nodes + yellow_nodes + blue_nodes)
        rgy_layer = {node.coordinate_2d: node for node in [*red_nodes, *green_nodes, *yellow_nodes]}
        rgy_layers[layer] = rgy_layer
        rgb_layer = {node.coordinate_2d: node for node in [*red_nodes, *green_nodes, *blue_nodes]}
        rgb_layers[layer] = rgb_layer

        # add nodes between layer and boundary nodes
        # left
        for offset, layer_dict in [(red_offset, rgy_layer), (green_offset, rgy_layer), (yellow_offset, rgb_layer)]:
            for position in triangular_node_position(layer, offset, only="left"):
                all_boundary_edges.append((left, layer_dict[position].pre_dg_node))
        # right
        for offset, layer_dict in [(red_offset, rgy_layer), (yellow_offset, rgy_layer), (yellow_offset, rgb_layer)]:
            for position in triangular_node_position(layer, offset, only="right"):
                all_boundary_edges.append((right, layer_dict[position].pre_dg_node))
        # center
        for offset, layer_dict in [(green_offset, rgy_layer), (yellow_offset, rgy_layer), (yellow_offset, rgb_layer)]:
            for position in triangular_node_position(layer, offset, only="center"):
                all_boundary_edges.append((center, layer_dict[position].pre_dg_node))

        # add edges between nodes of this layer
        for (x, y), node in rgy_layer.items():
            # blue nodes are placed on top of yellow ones, and share their edges
            if node.color == Color.yellow:
                all_edges.add((rgb_layer[(x, y)], node))
            # add trivial neighbours
            for x_offset in [-1, +1]:
                for y_offset in [-1, 0, +1]:
                    if n := rgy_layer.get((x+x_offset, y+y_offset)):
                        all_edges.add((node, n))
                        if node.color == Color.yellow:
                            all_edges.add((rgb_layer[(x, y)], n))
            # add neighbour above / below. If there is no node, add next-next neighbour
            if n := rgy_layer.get((x, y+1)):
                all_edges.add((node, n))
                if node.color == Color.yellow:
                    all_edges.add((rgb_layer[(x, y)], n))
            elif n := rgy_layer.get((x, y+2)):
                all_edges.add((node, n))
                if node.color == Color.yellow:
                    all_edges.add((rgb_layer[(x, y)], n))
            if n := rgy_layer.get((x, y-1)):
                all_edges.add((node, n))
                if node.color == Color.yellow:
                    all_edges.add((rgb_layer[(x, y)], n))
            elif n := rgy_layer.get((x, y-2)):
                all_edges.add((node, n))
                if node.color == Color.yellow:
                    all_edges.add((rgb_layer[(x, y)], n))

        # connect to bottom boundary
        if layer == num_layers:
            for node in rgy_layer.values():
                all_boundary_edges.append((bottom, node.pre_dg_node))
            continue

        # add edges between nodes of this and the lower layer
        for (x, y_), node in rgy_layer.items():
            # move nodes of this layer at the correct position at the lower layer
            if node.color == Color.red:
                y = y_ + 2
            else:
                y = y_ + 1

            # connect node with respective node of one level lower
            all_edges.add((rgb_layers[layer+1][(x, y)], node))

            if node.color == Color.green:
                continue

            # add trivial neighbours
            for x_offset in [-1, +1]:
                for y_offset in [-1, 0, +1]:
                    if (n := rgb_layers[layer+1].get((x+x_offset, y+y_offset))) and n.color != node.color:
                        all_edges.add((node, n))
            # add neighbour above / below. If there is no node, add next-next neighbour
            if (n := rgb_layers[layer+1].get((x, y+1))) and n.color != node.color:
                all_edges.add((node, n))
            elif (n := rgb_layers[layer+1].get((x, y+2))) and n.color != node.color:
                all_edges.add((node, n))
            if (n := rgb_layers[layer+1].get((x, y-1))) and n.color != node.color:
                all_edges.add((node, n))
            elif (n := rgb_layers[layer+1].get((x, y-2))) and n.color != node.color:
                all_edges.add((node, n))

    graph = rustworkx.PyGraph(multigraph=False)
    for node in [left, right, center, bottom]:
        index = graph.add_node(node)
        node.index = index
    for node in all_nodes:
        index = graph.add_node(node.pre_dg_node)
        node.index = index
    for edge in all_boundary_edges:
        add_edge(graph, edge[0], edge[1])
    for edge in all_edges:
        add_edge(graph, edge[0].pre_dg_node, edge[1].pre_dg_node)

    coloring_qubits(graph, 3, do_coloring=True)
    return graph

def cubic_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    """See https://www.nature.com/articles/ncomms12302#Sec12"""
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
    #              -111
    #
    # distance 4, top layer:
    #          |     |     |
    #       — 000 — 001 — 002 —
    #          |  /  |  \  |
    #       — 010 — 011 — 012 —
    #          |  \  |  /  |
    #       — 020 — 021 — 022 —
    #          |     |     |
    # distance 4, middle layer:
    #                 |
    #                1-11
    #           |  /  |  \  |
    #        — 100 — 101 — 102 —
    #        /  |  \  |  /  |  \
    # — 11-1 — 110 — 111 — 112 — 113 —
    #        \  |  /  |  \  |  /
    #        — 120 — 121 — 122 —
    #           |  \  |  /  |
    #                131
    #                 |
    # distance 4, bottom layer:
    #          |     |     |
    #       — 200 — 201 — 202 —
    #          |  /  |  \  |
    #       — 210 — 211 — 212 —
    #          |  \  |  /  |
    #       — 220 — 221 — 222 —
    #          |     |     |
    #
    #               311
    #
    # nodes = [[[000, 010, 020], [001, 011, 021], [002, 012, 022]],
    #          [[100, 110, 120], [102, 111, 121], [102, 112, 122]],
    #          [[200, 210, 220], [202, 211, 221], [202, 212, 222]]]
    #
    # face_nodes = [-111, 1-11, 11-1, 131, 113, 311]
    nodes = [[[PreDualGraphNode(f"({col},{row},{layer})")
               for row in range(num_rows)] for col in range(num_cols)] for layer in range(num_layers)]

    face_nodes: dict[tuple[int, int, int], PreDualGraphNode] = {}
    nodepos_to_facenodepos = {}
    facenodepos_to_nodepos = {}
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_cols):
                # one of the three indices is 0 or maximal (at the boundary)
                if (layer in {0, num_layers - 1}
                        # the other two indices are both odd
                        and row % 2 == col % 2 == 1
                        # and neither 0 nor maximal (at the boundary)
                        and row not in {0, num_rows - 1} and col not in {0, num_cols - 1}):
                    new_layer = -1 if layer == 0 else layer + 1
                    face_nodes[(col, row, new_layer)] = PreDualGraphNode(f"({col},{row},{new_layer})")
                    nodepos_to_facenodepos[(col, row, layer)] = (col, row, new_layer)
                    facenodepos_to_nodepos[(col, row, new_layer)] = (col, row, layer)
                # one of the three indices is 0 or maximal (at the boundary)
                elif (row in {0, num_rows - 1}
                        # the other two indices are both odd
                        and layer % 2 == col % 2 == 1
                        # and neither 0 nor maximal (at the boundary)
                        and layer not in {0, num_layers - 1} and col not in {0, num_cols - 1}):
                    new_row = -1 if row == 0 else row + 1
                    face_nodes[(col, new_row, layer)] = PreDualGraphNode(f"({col},{new_row},{layer})")
                    nodepos_to_facenodepos[(col, row, layer)] = (col, new_row, layer)
                    facenodepos_to_nodepos[(col, new_row, layer)] = (col, row, layer)
                # one of the three indices is 0 or maximal (at the boundary)
                elif (col in {0, num_cols - 1}
                        # the other two indices are both odd
                        and row % 2 == layer % 2 == 1
                        # and neither 0 nor maximal (at the boundary)
                        and row not in {0, num_rows - 1} and layer not in {0, num_layers - 1}):
                    new_col = -1 if col == 0 else col + 1
                    face_nodes[(new_col, row, layer)] = PreDualGraphNode(f"({new_col},{row},{layer})")
                    nodepos_to_facenodepos[(col, row, layer)] = (new_col, row, layer)
                    facenodepos_to_nodepos[(new_col, row, layer)] = (col, row, layer)

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for layer in nodes for row in layer for node in row if node is not None])
    dual_graph.add_nodes_from([face_node for face_node in face_nodes.values()])
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

    # between nodes/face_nodes and boundary_nodes
    for layer_pos, layer in enumerate(nodes):
        for col_pos, col in enumerate(layer):
            if face_node := face_nodes.get((col_pos, -1, layer_pos)):
                add_edge(dual_graph, face_node, front)
            else:
                add_edge(dual_graph, col[0], front)
            if face_node := face_nodes.get((col_pos, num_rows, layer_pos)):
                add_edge(dual_graph, face_node, back)
            else:
                add_edge(dual_graph, col[-1], back)
        # first row
        for row_pos, node in enumerate(layer[0]):
            if face_node := face_nodes.get((-1, row_pos, layer_pos)):
                add_edge(dual_graph, face_node, left)
            else:
                add_edge(dual_graph, node, left)
        # last row
        for row_pos, node in enumerate(layer[-1]):
            if face_node := face_nodes.get((num_cols, row_pos, layer_pos)):
                add_edge(dual_graph, face_node, right)
            else:
                add_edge(dual_graph, node, right)
    for col_pos, col in enumerate(nodes[0]):
        for row_pos, node in enumerate(col):
            if face_node := face_nodes.get((col_pos, row_pos, -1)):
                add_edge(dual_graph, face_node, top)
            else:
                add_edge(dual_graph, node, top)
    for col_pos, col in enumerate(nodes[-1]):
        for row_pos, node in enumerate(col):
            if face_node := face_nodes.get((col_pos, row_pos, num_layers)):
                add_edge(dual_graph, face_node, bottom)
            else:
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

    # between nodes and face_nodes
    for node_pos, face_node_pos in nodepos_to_facenodepos.items():
        face_node = face_nodes[face_node_pos]
        col_pos, row_pos, layer_pos = node_pos
        add_edge(dual_graph, nodes[layer_pos][col_pos][row_pos], face_node)
        if col_pos in {0, num_cols - 1}:
            add_edge(dual_graph, nodes[layer_pos - 1][col_pos][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos + 1][col_pos][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos][row_pos - 1], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos][row_pos + 1], face_node)
        elif row_pos in {0, num_rows - 1}:
            add_edge(dual_graph, nodes[layer_pos - 1][col_pos][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos + 1][col_pos][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos - 1][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos + 1][row_pos], face_node)
        elif layer_pos in {0, num_layers - 1}:
            add_edge(dual_graph, nodes[layer_pos][col_pos - 1][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos + 1][row_pos], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos][row_pos - 1], face_node)
            add_edge(dual_graph, nodes[layer_pos][col_pos][row_pos + 1], face_node)
        else:
            raise RuntimeError

    coloring_qubits(dual_graph, dimension=3)
    return dual_graph


def construct_x_dual_graph(dual_graph: rustworkx.PyGraph) -> rustworkx.PyGraph:
    """Where each edge of the dual_graph is a node in the x_dual_graph.

    An edge is added between two nodes (= faces of the primary graph) if they share at least one qubit.
    """
    max_id = 0
    nodes = []
    for edge in dual_graph.edges():
        color = edge.node1.color.combine(edge.node2.color)
        node = XDualGraphNode(max_id, color, edge.qubits, edge.is_stabilizer, edge.all_qubits)
        max_id += 1
        nodes.append(node)

    x_dual_graph = rustworkx.PyGraph(multigraph=False)
    x_dual_graph.add_nodes_from(nodes)
    for index in x_dual_graph.node_indices():
        x_dual_graph[index].index = index

    # insert edges between the nodes
    for node1, node2 in itertools.combinations(x_dual_graph.nodes(), 2):
        if x_dual_graph.has_edge(node1.index, node2.index):
            continue
        # elif node1.is_boundary and node2.is_boundary:
        #     continue
        elif set(node1.qubits) & set(node2.qubits):
            x_dual_graph.add_edge(node1.index, node2.index, XDualGraphEdge(max_id, node1, node2))
            max_id += 1
    for index in x_dual_graph.edge_indices():
        x_dual_graph.edge_index_map()[index][2].index = index

    return x_dual_graph


def construct_cubic_logicals(dual_graph: rustworkx.PyGraph) -> tuple[list[Operator], list[Operator]]:
    """Construct the x and z logical operators from the dual graph of a 3D cubic color code."""
    boundary_nodes: list[DualGraphNode] = [node for node in dual_graph.nodes() if node.is_boundary]
    boundary_nodes_by_color: dict[Color, tuple[DualGraphNode, DualGraphNode]] = {
        node1.color: (node1, node2) for node1, node2 in itertools.combinations(boundary_nodes, 2) if
        node1.color == node2.color
    }
    if len(boundary_nodes_by_color) != 3:
        raise ValueError

    # map x operators to z operators
    logicals: dict[Operator, Operator] = {}
    for color, (node1, node2) in boundary_nodes_by_color.items():
        # faces of the cube
        x_logical = Operator(len(node1.all_qubits), x_positions=node1.qubits)

        # edges of the cube
        other_boundaries = []
        for col in Color.get_monochrome():
            if col == color:
                continue
            if col not in boundary_nodes_by_color:
                continue
            other_boundaries.append(boundary_nodes_by_color[col][0])
        support = list(set(other_boundaries[0].qubits) & set(other_boundaries[1].qubits))
        logicals[x_logical] = Operator(len(node1.all_qubits), z_positions=support)

    x_logicals = sorted(logicals.keys())
    z_logicals = [logicals[x_logical] for x_logical in x_logicals]

    return x_logicals, z_logicals


def construct_tetrahedron_logicals(dual_graph: rustworkx.PyGraph) -> tuple[list[Operator], list[Operator]]:
    boundary_nodes: list[DualGraphNode] = [node for node in dual_graph.nodes() if node.is_boundary]
    x_face = boundary_nodes[0].qubits
    z_edge = list(set(boundary_nodes[1].qubits) & set(boundary_nodes[2].qubits))
    all_qubits = boundary_nodes[0].all_qubits

    x_logicals = [Operator(len(all_qubits), x_positions=x_face)]
    z_logicals = [Operator(len(all_qubits), z_positions=z_edge)]

    return x_logicals, z_logicals
