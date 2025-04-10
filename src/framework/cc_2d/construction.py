"""Construct dual graph representations of rectangular and square 2D color codes."""

import itertools

import rustworkx

from framework.common.construction import PreDualGraphNode, add_edge, coloring_qubits


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
