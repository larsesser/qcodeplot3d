import itertools

import rustworkx

from framework.base import DualGraphNode, XDualGraphEdge, XDualGraphNode
from framework.construction import PreDualGraphNode, add_edge, coloring_qubits
from framework.stabilizers import Color, Operator


def tetrahedron_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    if distance != 3:
        raise NotImplementedError

    dual_graph = rustworkx.PyGraph(multigraph=False)
    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    back = PreDualGraphNode("back", is_boundary=True)
    front = PreDualGraphNode("front", is_boundary=True)
    boundaries = [left, right, back, front]

    nodes = [PreDualGraphNode("1"), PreDualGraphNode("2"), PreDualGraphNode("3"), PreDualGraphNode("4")]

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from(nodes)
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # edges between boundary and boundary
    for node1, node2 in itertools.combinations(boundaries, 2):
        add_edge(dual_graph, node1, node2)

    # edges between boundary and node
    for excluded_index, boundary in enumerate(boundaries):
        for index, node in enumerate(nodes):
            if excluded_index == index:
                continue
            add_edge(dual_graph, boundary, node)

    # edges between node and node
    for node1, node2 in itertools.combinations(nodes, 2):
        add_edge(dual_graph, node1, node2)

    coloring_qubits(dual_graph, dimension=3)
    return dual_graph


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


def cubic_3d_d4_dual_graph(distance: int) -> rustworkx.PyGraph:
    """See https://www.nature.com/articles/ncomms12302#Sec12"""
    raise NotImplementedError

    dual_graph = rustworkx.PyGraph(multigraph=False)

    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    back = PreDualGraphNode("back", is_boundary=True)
    front = PreDualGraphNode("front", is_boundary=True)
    top = PreDualGraphNode("top", is_boundary=True)
    bottom = PreDualGraphNode("bottom", is_boundary=True)
    boundaries = [left, right, back, front, top, bottom]

    # distance 4, top layer:
    #    |     |     |
    # – 000 - 001 - 002 –
    #    |  \  |  /  |
    # – 010 – 011 - 012 -
    #    |  /  |  \  |
    # – 020 - 021 - 022 -
    #    |     |     |
    # distance 4, middle layer:
    #    |     |     |
    # – 100 - 101 - 102 –
    #    |  /  |  \  |
    # – 110 – 111 - 112 -
    #    |  \  |  /  |
    # – 120 - 121 - 122 -
    #    |     |     |
    # distance 4, bottom layer:
    #    |     |     |
    # – 200 - 201 - 202 –
    #    |  \  |  /  |
    # – 210 – 211 - 212 -
    #    |  /  |  \  |
    # – 220 - 221 - 222 -
    #    |     |     |

    node_000 = PreDualGraphNode("000")
    node_001 = PreDualGraphNode("001")
    node_002 = PreDualGraphNode("002")
    node_010 = PreDualGraphNode("010")
    node_011 = PreDualGraphNode("011")
    node_012 = PreDualGraphNode("012")
    node_020 = PreDualGraphNode("020")
    node_021 = PreDualGraphNode("021")
    node_022 = PreDualGraphNode("022")
    node_100 = PreDualGraphNode("100")
    node_101 = PreDualGraphNode("101")
    node_102 = PreDualGraphNode("102")
    node_110 = PreDualGraphNode("110")
    node_111 = PreDualGraphNode("111")
    node_112 = PreDualGraphNode("112")
    node_120 = PreDualGraphNode("120")
    node_121 = PreDualGraphNode("121")
    node_122 = PreDualGraphNode("122")
    node_200 = PreDualGraphNode("200")
    node_201 = PreDualGraphNode("201")
    node_202 = PreDualGraphNode("202")
    node_210 = PreDualGraphNode("210")
    node_211 = PreDualGraphNode("211")
    node_212 = PreDualGraphNode("212")
    node_220 = PreDualGraphNode("220")
    node_221 = PreDualGraphNode("221")
    node_222 = PreDualGraphNode("222")

    nodes = [node_000, node_001, node_002,
             node_010, node_011, node_012,
             node_020, node_021, node_022,
             node_100, node_101, node_102,
             node_110, node_111, node_112,
             node_120, node_121, node_122,
             node_200, node_201, node_202,
             node_210, node_211, node_212,
             node_220, node_221, node_222]

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from(nodes)
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # boundaries with boundaries
    add_edge(dual_graph, left, back)
    add_edge(dual_graph, back, right)
    add_edge(dual_graph, right, front)
    add_edge(dual_graph, front, left)
    for node in [left, right, back, front]:
        add_edge(dual_graph, top, node)
        add_edge(dual_graph, bottom, node)

    # nodes with boundaries
    for node in [node_000, node_010, node_020,
                 node_100, node_110, node_120,
                 node_200, node_210, node_220]:
        add_edge(dual_graph, node, left)
    for node in [node_002, node_012, node_022,
                 node_102, node_112, node_122,
                 node_202, node_212, node_222]:
        add_edge(dual_graph, node, right)
    for node in [node_000, node_001, node_002,
                 node_100, node_101, node_102,
                 node_200, node_201, node_202]:
        add_edge(dual_graph, node, back)
    for node in [node_020, node_021, node_022,
                 node_120, node_121, node_122,
                 node_220, node_221, node_222]:
        add_edge(dual_graph, node, front)
    for node in [node_000, node_001, node_002,
                 node_010, node_011, node_012,
                 node_020, node_021, node_022]:
        add_edge(dual_graph, node, top)
    for node in [node_200, node_201, node_202,
                 node_210, node_211, node_212,
                 node_220, node_221, node_222]:
        add_edge(dual_graph, node, bottom)

    # nodes back-front
    add_edge(dual_graph, node_000, node_010)
    add_edge(dual_graph, node_010, node_020)
    add_edge(dual_graph, node_001, node_011)
    add_edge(dual_graph, node_011, node_021)
    add_edge(dual_graph, node_002, node_012)
    add_edge(dual_graph, node_012, node_022)

    add_edge(dual_graph, node_100, node_110)
    add_edge(dual_graph, node_110, node_120)
    add_edge(dual_graph, node_101, node_111)
    add_edge(dual_graph, node_111, node_121)
    add_edge(dual_graph, node_102, node_112)
    add_edge(dual_graph, node_112, node_122)

    add_edge(dual_graph, node_200, node_210)
    add_edge(dual_graph, node_210, node_220)
    add_edge(dual_graph, node_201, node_211)
    add_edge(dual_graph, node_211, node_221)
    add_edge(dual_graph, node_202, node_212)
    add_edge(dual_graph, node_212, node_222)

    # nodes left-right
    add_edge(dual_graph, node_000, node_001)
    add_edge(dual_graph, node_001, node_002)
    add_edge(dual_graph, node_010, node_011)
    add_edge(dual_graph, node_011, node_012)
    add_edge(dual_graph, node_020, node_021)
    add_edge(dual_graph, node_021, node_022)

    add_edge(dual_graph, node_100, node_101)
    add_edge(dual_graph, node_101, node_102)
    add_edge(dual_graph, node_110, node_111)
    add_edge(dual_graph, node_111, node_112)
    add_edge(dual_graph, node_120, node_121)
    add_edge(dual_graph, node_121, node_122)

    add_edge(dual_graph, node_200, node_201)
    add_edge(dual_graph, node_201, node_202)
    add_edge(dual_graph, node_210, node_211)
    add_edge(dual_graph, node_211, node_212)
    add_edge(dual_graph, node_220, node_221)
    add_edge(dual_graph, node_221, node_222)

    # nodes top-bottom (first, second, third row)
    add_edge(dual_graph, node_000, node_100)
    add_edge(dual_graph, node_100, node_200)
    add_edge(dual_graph, node_001, node_101)
    add_edge(dual_graph, node_101, node_201)
    add_edge(dual_graph, node_002, node_102)
    add_edge(dual_graph, node_102, node_202)

    add_edge(dual_graph, node_010, node_110)
    add_edge(dual_graph, node_110, node_210)
    add_edge(dual_graph, node_011, node_111)
    add_edge(dual_graph, node_111, node_211)
    add_edge(dual_graph, node_012, node_112)
    add_edge(dual_graph, node_112, node_212)

    add_edge(dual_graph, node_020, node_120)
    add_edge(dual_graph, node_120, node_220)
    add_edge(dual_graph, node_021, node_121)
    add_edge(dual_graph, node_121, node_221)
    add_edge(dual_graph, node_022, node_122)
    add_edge(dual_graph, node_122, node_222)

    # nodes diagonal (first, second, third layer)
    add_edge(dual_graph, node_000, node_011)
    add_edge(dual_graph, node_002, node_011)
    add_edge(dual_graph, node_020, node_011)
    add_edge(dual_graph, node_022, node_011)

    add_edge(dual_graph, node_110, node_121)
    add_edge(dual_graph, node_121, node_112)
    add_edge(dual_graph, node_112, node_101)
    add_edge(dual_graph, node_101, node_110)

    add_edge(dual_graph, node_200, node_211)
    add_edge(dual_graph, node_202, node_211)
    add_edge(dual_graph, node_220, node_211)
    add_edge(dual_graph, node_222, node_211)

    # nodes diagonal between layers
    add_edge(dual_graph, node_000, node_110)
    add_edge(dual_graph, node_020, node_110)
    add_edge(dual_graph, node_200, node_110)
    add_edge(dual_graph, node_220, node_110)

    add_edge(dual_graph, node_020, node_121)
    add_edge(dual_graph, node_022, node_121)
    add_edge(dual_graph, node_220, node_121)
    add_edge(dual_graph, node_222, node_121)

    add_edge(dual_graph, node_022, node_112)
    add_edge(dual_graph, node_002, node_112)
    add_edge(dual_graph, node_222, node_112)
    add_edge(dual_graph, node_202, node_112)

    add_edge(dual_graph, node_000, node_101)
    add_edge(dual_graph, node_002, node_101)
    add_edge(dual_graph, node_200, node_101)
    add_edge(dual_graph, node_202, node_101)

    add_edge(dual_graph, node_110, node_011)
    add_edge(dual_graph, node_121, node_011)
    add_edge(dual_graph, node_112, node_011)
    add_edge(dual_graph, node_101, node_011)

    add_edge(dual_graph, node_110, node_211)
    add_edge(dual_graph, node_121, node_211)
    add_edge(dual_graph, node_112, node_211)
    add_edge(dual_graph, node_101, node_211)

    coloring_qubits(dual_graph, dimension=3, do_coloring=False)
    return dual_graph


def cubic_3d_d4_2_dual_graph(distance: int) -> rustworkx.PyGraph:
    """See https://www.nature.com/articles/ncomms12302#Sec12"""
    if not distance != 4:
        raise NotImplementedError

    dual_graph = rustworkx.PyGraph(multigraph=False)

    left = PreDualGraphNode("left", is_boundary=True)
    right = PreDualGraphNode("right", is_boundary=True)
    back = PreDualGraphNode("back", is_boundary=True)
    front = PreDualGraphNode("front", is_boundary=True)
    top = PreDualGraphNode("top", is_boundary=True)
    bottom = PreDualGraphNode("bottom", is_boundary=True)
    boundaries = [left, right, back, front, top, bottom]

    #              911
    #
    # distance 4, top layer:
    #          |     |     |
    #       – 000 - 001 - 002 –
    #          |  /  |  \  |
    #       – 010 – 011 - 012 -
    #          |  \  |  /  |
    #       – 020 - 021 - 022 -
    #          |     |     |
    # distance 4, middle layer:
    #                |
    #               191
    #          |  /  |  \  |
    #       – 100 - 101 - 102 –
    #       /  |  \  |  /  |  \
    # - 119 – 110 – 111 - 112 - 113 -
    #       \  |  /  |  \  |  /
    #       – 120 - 121 - 122 -
    #          |  \  |  /  |
    #               131
    #                |
    # distance 4, bottom layer:
    #          |     |     |
    #       – 200 - 201 - 202 –
    #          |  /  |  \  |
    #       – 210 – 211 - 212 -
    #          |  \  |  /  |
    #       – 220 - 221 - 222 -
    #          |     |     |
    #
    #               311

    node_000 = PreDualGraphNode("000")
    node_001 = PreDualGraphNode("001")
    node_002 = PreDualGraphNode("002")
    node_010 = PreDualGraphNode("010")
    node_011 = PreDualGraphNode("011")
    node_012 = PreDualGraphNode("012")
    node_020 = PreDualGraphNode("020")
    node_021 = PreDualGraphNode("021")
    node_022 = PreDualGraphNode("022")
    node_100 = PreDualGraphNode("100")
    node_101 = PreDualGraphNode("101")
    node_102 = PreDualGraphNode("102")
    node_110 = PreDualGraphNode("110")
    node_111 = PreDualGraphNode("111")
    node_112 = PreDualGraphNode("112")
    node_120 = PreDualGraphNode("120")
    node_121 = PreDualGraphNode("121")
    node_122 = PreDualGraphNode("122")
    node_200 = PreDualGraphNode("200")
    node_201 = PreDualGraphNode("201")
    node_202 = PreDualGraphNode("202")
    node_210 = PreDualGraphNode("210")
    node_211 = PreDualGraphNode("211")
    node_212 = PreDualGraphNode("212")
    node_220 = PreDualGraphNode("220")
    node_221 = PreDualGraphNode("221")
    node_222 = PreDualGraphNode("222")

    node_911 = PreDualGraphNode("911")
    node_119 = PreDualGraphNode("119")
    node_131 = PreDualGraphNode("131")
    node_113 = PreDualGraphNode("113")
    node_191 = PreDualGraphNode("191")
    node_311 = PreDualGraphNode("311")

    nodes = [node_000, node_001, node_002,
             node_010, node_011, node_012,
             node_020, node_021, node_022,
             node_100, node_101, node_102,
             node_110, node_111, node_112,
             node_120, node_121, node_122,
             node_200, node_201, node_202,
             node_210, node_211, node_212,
             node_220, node_221, node_222,
             node_911, node_119, node_131,
             node_113, node_191, node_311]

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from(nodes)
    for index in dual_graph.node_indices():
        dual_graph[index].index = index

    # boundaries with boundaries
    add_edge(dual_graph, left, back)
    add_edge(dual_graph, back, right)
    add_edge(dual_graph, right, front)
    add_edge(dual_graph, front, left)
    for node in [left, right, back, front]:
        add_edge(dual_graph, top, node)
        add_edge(dual_graph, bottom, node)

    # nodes with boundaries
    for node in [node_000, node_010, node_020,
                 node_100, node_119, node_120,
                 node_200, node_210, node_220]:
        add_edge(dual_graph, node, left)
    for node in [node_002, node_012, node_022,
                 node_102, node_113, node_122,
                 node_202, node_212, node_222]:
        add_edge(dual_graph, node, right)
    for node in [node_000, node_001, node_002,
                 node_100, node_191, node_102,
                 node_200, node_201, node_202]:
        add_edge(dual_graph, node, back)
    for node in [node_020, node_021, node_022,
                 node_120, node_131, node_122,
                 node_220, node_221, node_222]:
        add_edge(dual_graph, node, front)
    for node in [node_000, node_001, node_002,
                 node_010, node_911, node_012,
                 node_020, node_021, node_022]:
        add_edge(dual_graph, node, top)
    for node in [node_200, node_201, node_202,
                 node_210, node_311, node_212,
                 node_220, node_221, node_222]:
        add_edge(dual_graph, node, bottom)

    # nodes back-front
    add_edge(dual_graph, node_000, node_010)
    add_edge(dual_graph, node_010, node_020)
    add_edge(dual_graph, node_001, node_011)
    add_edge(dual_graph, node_011, node_021)
    add_edge(dual_graph, node_002, node_012)
    add_edge(dual_graph, node_012, node_022)

    add_edge(dual_graph, node_100, node_110)
    add_edge(dual_graph, node_110, node_120)
    add_edge(dual_graph, node_101, node_111)
    add_edge(dual_graph, node_111, node_121)
    add_edge(dual_graph, node_102, node_112)
    add_edge(dual_graph, node_112, node_122)

    add_edge(dual_graph, node_200, node_210)
    add_edge(dual_graph, node_210, node_220)
    add_edge(dual_graph, node_201, node_211)
    add_edge(dual_graph, node_211, node_221)
    add_edge(dual_graph, node_202, node_212)
    add_edge(dual_graph, node_212, node_222)

    # nodes left-right
    add_edge(dual_graph, node_000, node_001)
    add_edge(dual_graph, node_001, node_002)
    add_edge(dual_graph, node_010, node_011)
    add_edge(dual_graph, node_011, node_012)
    add_edge(dual_graph, node_020, node_021)
    add_edge(dual_graph, node_021, node_022)

    add_edge(dual_graph, node_100, node_101)
    add_edge(dual_graph, node_101, node_102)
    add_edge(dual_graph, node_110, node_111)
    add_edge(dual_graph, node_111, node_112)
    add_edge(dual_graph, node_120, node_121)
    add_edge(dual_graph, node_121, node_122)

    add_edge(dual_graph, node_200, node_201)
    add_edge(dual_graph, node_201, node_202)
    add_edge(dual_graph, node_210, node_211)
    add_edge(dual_graph, node_211, node_212)
    add_edge(dual_graph, node_220, node_221)
    add_edge(dual_graph, node_221, node_222)

    # nodes top-bottom (first, second, third row)
    add_edge(dual_graph, node_000, node_100)
    add_edge(dual_graph, node_100, node_200)
    add_edge(dual_graph, node_001, node_101)
    add_edge(dual_graph, node_101, node_201)
    add_edge(dual_graph, node_002, node_102)
    add_edge(dual_graph, node_102, node_202)

    add_edge(dual_graph, node_010, node_110)
    add_edge(dual_graph, node_110, node_210)
    add_edge(dual_graph, node_011, node_111)
    add_edge(dual_graph, node_111, node_211)
    add_edge(dual_graph, node_012, node_112)
    add_edge(dual_graph, node_112, node_212)

    add_edge(dual_graph, node_020, node_120)
    add_edge(dual_graph, node_120, node_220)
    add_edge(dual_graph, node_021, node_121)
    add_edge(dual_graph, node_121, node_221)
    add_edge(dual_graph, node_022, node_122)
    add_edge(dual_graph, node_122, node_222)

    # nodes diagonal (first, second, third layer)
    add_edge(dual_graph, node_010, node_021)
    add_edge(dual_graph, node_021, node_012)
    add_edge(dual_graph, node_012, node_001)
    add_edge(dual_graph, node_001, node_010)

    add_edge(dual_graph, node_100, node_111)
    add_edge(dual_graph, node_102, node_111)
    add_edge(dual_graph, node_120, node_111)
    add_edge(dual_graph, node_122, node_111)

    add_edge(dual_graph, node_210, node_221)
    add_edge(dual_graph, node_221, node_212)
    add_edge(dual_graph, node_212, node_201)
    add_edge(dual_graph, node_201, node_210)

    # nodes diagonal between layers
    add_edge(dual_graph, node_001, node_100)
    add_edge(dual_graph, node_010, node_100)
    add_edge(dual_graph, node_010, node_120)
    add_edge(dual_graph, node_021, node_120)
    add_edge(dual_graph, node_021, node_122)
    add_edge(dual_graph, node_012, node_122)
    add_edge(dual_graph, node_012, node_102)
    add_edge(dual_graph, node_001, node_102)

    add_edge(dual_graph, node_001, node_111)
    add_edge(dual_graph, node_010, node_111)
    add_edge(dual_graph, node_021, node_111)
    add_edge(dual_graph, node_012, node_111)

    add_edge(dual_graph, node_201, node_100)
    add_edge(dual_graph, node_210, node_100)
    add_edge(dual_graph, node_210, node_120)
    add_edge(dual_graph, node_221, node_120)
    add_edge(dual_graph, node_221, node_122)
    add_edge(dual_graph, node_212, node_122)
    add_edge(dual_graph, node_212, node_102)
    add_edge(dual_graph, node_201, node_102)

    add_edge(dual_graph, node_201, node_111)
    add_edge(dual_graph, node_210, node_111)
    add_edge(dual_graph, node_221, node_111)
    add_edge(dual_graph, node_212, node_111)

    # special nodes
    # left
    add_edge(dual_graph, node_100, node_119)
    add_edge(dual_graph, node_110, node_119)
    add_edge(dual_graph, node_120, node_119)
    add_edge(dual_graph, node_010, node_119)
    add_edge(dual_graph, node_210, node_119)
    # front
    add_edge(dual_graph, node_120, node_131)
    add_edge(dual_graph, node_121, node_131)
    add_edge(dual_graph, node_122, node_131)
    add_edge(dual_graph, node_021, node_131)
    add_edge(dual_graph, node_221, node_131)
    # right
    add_edge(dual_graph, node_102, node_113)
    add_edge(dual_graph, node_112, node_113)
    add_edge(dual_graph, node_122, node_113)
    add_edge(dual_graph, node_012, node_113)
    add_edge(dual_graph, node_212, node_113)
    # back
    add_edge(dual_graph, node_100, node_191)
    add_edge(dual_graph, node_101, node_191)
    add_edge(dual_graph, node_102, node_191)
    add_edge(dual_graph, node_001, node_191)
    add_edge(dual_graph, node_201, node_191)
    # bottom
    add_edge(dual_graph, node_201, node_311)
    add_edge(dual_graph, node_210, node_311)
    add_edge(dual_graph, node_221, node_311)
    add_edge(dual_graph, node_212, node_311)
    add_edge(dual_graph, node_211, node_311)
    # top
    add_edge(dual_graph, node_001, node_911)
    add_edge(dual_graph, node_010, node_911)
    add_edge(dual_graph, node_021, node_911)
    add_edge(dual_graph, node_012, node_911)
    add_edge(dual_graph, node_011, node_911)

    coloring_qubits(dual_graph, dimension=3, do_coloring=True)
    return dual_graph


def cubic2_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    """Magic-boundary lattice, biggest volume (c1) in center."""
    # TODO: Ecken anders abschneiden? Sodass sie nicht zu beiden boundaries gehören, dafür aber nicht weight-5 kanten produzieren?
    raise NotImplementedError
    if not distance % 2 == 0:
        raise ValueError("d must be an even integer")

    if not distance == 4:
        raise NotImplementedError

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
    #   | \ | / |
    # – d – e - f -
    #   | / | \ |
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
    #   | \ | / |
    # – v – w - x -
    #   | / | \ |
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
                # diagonal pattern
                if not ((row_pos % 2) == 1 and (col_pos % 2) == 1):
                    continue
                add_edge(dual_graph, node, layer[col_pos+1][row_pos+1])
                add_edge(dual_graph, node, layer[col_pos+1][row_pos-1])
                add_edge(dual_graph, node, layer[col_pos-1][row_pos+1])
                add_edge(dual_graph, node, layer[col_pos-1][row_pos-1])
    # between two layers
    for layer1, layer2 in zip(nodes, nodes[1:]):
        for col1, col2 in zip(layer1, layer2):
            for node1, node2 in zip(col1, col2):
                add_edge(dual_graph, node1, node2)

    layer0 = nodes[0]
    layer1 = nodes[1]
    layer2 = nodes[2]
    for layer in [layer0, layer2]:
        add_edge(dual_graph, layer1[1][0], layer[0][0])
        add_edge(dual_graph, layer1[1][0], layer[2][0])

        add_edge(dual_graph, layer1[2][1], layer[2][0])
        add_edge(dual_graph, layer1[2][1], layer[2][2])

        add_edge(dual_graph, layer1[1][2], layer[2][2])
        add_edge(dual_graph, layer1[1][2], layer[0][2])

        add_edge(dual_graph, layer1[0][1], layer[0][2])
        add_edge(dual_graph, layer1[0][1], layer[0][0])

        for node in itertools.chain.from_iterable(layer):
            add_edge(dual_graph, node, layer1[1][1])

    coloring_qubits(dual_graph, dimension=3, do_coloring=False)
    return dual_graph


def cubic3_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    """Magic-boundary lattice, smallest volume (c3) in center."""
    raise NotImplementedError
    if not distance % 2 == 0:
        raise ValueError("d must be an even integer")

    if not distance == 4:
        raise NotImplementedError

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
    #       |
    #       b
    #       |
    # – d – e - f -
    #       |
    #       h
    #       |
    # distance 4, middle layer:
    #   |       |
    # – j - - - l –
    #   | \   / |
    #   |   n   |
    #   | /   \ |
    # – p - - - r -
    #   |       |
    # distance 4, bottom layer:
    #       |
    #       b
    #       |
    # – d – e - f -
    #       |
    #       h
    #       |
    nodes = [[[PreDualGraphNode(f"({col},{row},{layer})")
               for row in range(num_rows)] for col in range(num_cols)] for layer in range(num_layers)]
    for layer_pos, layer in enumerate(nodes):
        for row_pos, row in enumerate(layer):
            for col_pos, node in enumerate(row):
                if (layer_pos % 2 == 0) and (row_pos % 2 != col_pos % 2 or row_pos == col_pos == 1):
                    continue
                if (layer_pos % 2 == 1) and (row_pos % 2 == col_pos % 2):
                    continue
                nodes[layer_pos][row_pos][col_pos] = None

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for layer in nodes for row in layer for node in row if node is not None])
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

    layer0 = nodes[0]
    layer1 = nodes[1]
    layer2 = nodes[2]

    # inside one layer
    for layer in [layer0, layer2]:
        add_edge(dual_graph, layer[0][1], layer[1][1])
        add_edge(dual_graph, layer[1][0], layer[1][1])
        add_edge(dual_graph, layer[2][1], layer[1][1])
        add_edge(dual_graph, layer[1][2], layer[1][1])

    add_edge(dual_graph, layer1[0][0], layer1[0][2])
    add_edge(dual_graph, layer1[0][2], layer1[2][2])
    add_edge(dual_graph, layer1[2][2], layer1[2][0])
    add_edge(dual_graph, layer1[2][0], layer1[0][0])
    add_edge(dual_graph, layer1[0][0], layer1[1][1])
    add_edge(dual_graph, layer1[0][2], layer1[1][1])
    add_edge(dual_graph, layer1[2][2], layer1[1][1])
    add_edge(dual_graph, layer1[2][0], layer1[1][1])

    # between layers
    for layer in [layer0, layer2]:
        add_edge(dual_graph, layer[0][1], layer1[0][0])
        add_edge(dual_graph, layer[1][0], layer1[0][0])

        add_edge(dual_graph, layer[0][1], layer1[0][2])
        add_edge(dual_graph, layer[1][2], layer1[0][2])

        add_edge(dual_graph, layer[1][2], layer1[2][2])
        add_edge(dual_graph, layer[2][1], layer1[2][2])

        add_edge(dual_graph, layer[2][1], layer1[2][0])
        add_edge(dual_graph, layer[1][0], layer1[2][0])
    for node in itertools.chain.from_iterable(layer1):
        add_edge(dual_graph, node, layer0[1][1])
        add_edge(dual_graph, node, layer2[1][1])

    coloring_qubits(dual_graph, dimension=3, do_coloring=False)
    return dual_graph


def cubic4_3d_dual_graph(distance: int) -> rustworkx.PyGraph:
    """Magic-boundary lattice, medium volume (c2) in center."""
    raise NotImplementedError
    if not distance % 2 == 0:
        raise ValueError("d must be an even integer")

    if not distance == 4:
        raise NotImplementedError

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
    #   |       |
    # – j - - - l –
    #   | \   / |
    #   |   n   |
    #   | /   \ |
    # – p - - - r -
    #   |       |
    # distance 4, middle layer:
    #       |
    #       b
    #       |
    # – d – e - f -
    #       |
    #       h
    #       |
    # distance 4, bottom layer:
    #   |       |
    # – j - - - l –
    #   | \   / |
    #   |   n   |
    #   | /   \ |
    # – p - - - r -
    #   |       |
    nodes = [[[PreDualGraphNode(f"({col},{row},{layer})")
               for row in range(num_rows)] for col in range(num_cols)] for layer in range(num_layers)]
    for layer_pos, layer in enumerate(nodes):
        for row_pos, row in enumerate(layer):
            for col_pos, node in enumerate(row):
                if (layer_pos % 2 == 1) and (row_pos % 2 != col_pos % 2 or row_pos == col_pos == 1):
                    continue
                if (layer_pos % 2 == 0) and (row_pos % 2 == col_pos % 2):
                    continue
                nodes[layer_pos][row_pos][col_pos] = None

    dual_graph.add_nodes_from(boundaries)
    dual_graph.add_nodes_from([node for layer in nodes for row in layer for node in row if node is not None])
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

    layer0 = nodes[0]
    layer1 = nodes[1]
    layer2 = nodes[2]

    # inside one layer
    add_edge(dual_graph, layer1[0][1], layer1[1][1])
    add_edge(dual_graph, layer1[1][0], layer1[1][1])
    add_edge(dual_graph, layer1[2][1], layer1[1][1])
    add_edge(dual_graph, layer1[1][2], layer1[1][1])

    for layer in [layer0, layer2]:
        add_edge(dual_graph, layer[0][0], layer[0][2])
        add_edge(dual_graph, layer[0][2], layer[2][2])
        add_edge(dual_graph, layer[2][2], layer[2][0])
        add_edge(dual_graph, layer[2][0], layer[0][0])
        add_edge(dual_graph, layer[0][0], layer[1][1])
        add_edge(dual_graph, layer[0][2], layer[1][1])
        add_edge(dual_graph, layer[2][2], layer[1][1])
        add_edge(dual_graph, layer[2][0], layer[1][1])

    # between layers
    for layer in [layer0, layer2]:
        add_edge(dual_graph, layer1[0][1], layer[0][0])
        add_edge(dual_graph, layer1[1][0], layer[0][0])

        add_edge(dual_graph, layer1[0][1], layer[0][2])
        add_edge(dual_graph, layer1[1][2], layer[0][2])

        add_edge(dual_graph, layer1[1][2], layer[2][2])
        add_edge(dual_graph, layer1[2][1], layer[2][2])

        add_edge(dual_graph, layer1[2][1], layer[2][0])
        add_edge(dual_graph, layer1[1][0], layer[2][0])
    for layer in [layer0, layer2]:
        for node in itertools.chain.from_iterable(layer):
            add_edge(dual_graph, node, layer1[1][1])
    add_edge(dual_graph, layer0[0][0], layer2[0][0])
    add_edge(dual_graph, layer0[0][2], layer2[0][2])
    add_edge(dual_graph, layer0[2][2], layer2[2][2])
    add_edge(dual_graph, layer0[2][0], layer2[2][0])

    coloring_qubits(dual_graph, dimension=3, do_coloring=False)
    return dual_graph


def construct_x_dual_graph(dual_graph: rustworkx.PyGraph) -> rustworkx.PyGraph:
    """Where each edge of the dual_graph is a node in the x_dual_graph."""
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

    x_logicals = []
    z_logicals = []
    for color, (node1, node2) in boundary_nodes_by_color.items():
        # faces of the cube
        x_logicals.append(Operator(len(node1.all_qubits), x_positions=node1.qubits))

        # edges of the cube
        other_boundaries = []
        for col in Color.get_monochrome():
            if col == color:
                continue
            if col not in boundary_nodes_by_color:
                continue
            other_boundaries.append(boundary_nodes_by_color[col][0])
        support = list(set(other_boundaries[0].qubits) & set(other_boundaries[1].qubits))
        z_logicals.append(Operator(len(node1.all_qubits), z_positions=support))

    return x_logicals, z_logicals
