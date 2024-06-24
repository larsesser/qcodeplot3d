import collections

import rustworkx

from framework.decoder import ConcatenatedDecoder, GraphNode, GraphEdge, DualGraphNode, DualGraphEdge
from rustworkx.visualization import graphviz_draw
from framework.stabilizers import Color, Stabilizer
from framework.layer import Syndrome, SyndromeValue
from copy import deepcopy
import rustworkx as rx
import itertools


class PreDualGraphNode(GraphNode):
    def __init__(self, is_boundary: bool = False):
        self._is_boundary = is_boundary

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

    # In the following, we construct all necessary information from the graph layout:
    # - color of the nodes
    # - adjacent qubits to stabilizers / boundaries

    # rustworkx guarantees that the undirected and directed graph share the same indexes for the same objects.
    directed_dual_graph = dual_graph.to_directed()

    # compute a coloring of the graph
    coloring = rustworkx.graph_greedy_color(dual_graph)
    colorints = list(set(coloring.values()))
    if len(colorints) != 4:
        raise ValueError(colorints)
    colorint2color = {colorints[0]: Color.red, colorints[1]: Color.blue, colorints[2]: Color.green, colorints[3]: Color.yellow}
    coloring = {node_index: colorint2color[colorint] for node_index, colorint in coloring.items()}

    # find all tetrahedrons of the graph
    cycles = {tuple(sorted(indices)) for indices in rx.simple_cycles(directed_dual_graph) if len(indices) == 3}
    tetrahedrons_set = set()
    for cycle in cycles:
        common_neighbours = set(dual_graph.neighbors(cycle[0])) & set(dual_graph.neighbors(cycle[1])) & set(dual_graph.neighbors(cycle[2]))
        for neighbour in common_neighbours:
            tetrahedrons_set.add(tuple(sorted([*cycle, neighbour])))
    tetrahedrons = {tetrahedron: number for number, tetrahedron in enumerate(tetrahedrons_set, start=1)}
    node2tetrahedron = collections.defaultdict(list)
    for indices, tetrahedron in tetrahedrons.items():
        for index in indices:
            node2tetrahedron[index].append(tetrahedron)

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

    return dual_graph


def node_attr_fn(node: GraphNode):
    label = f"{node.id}, {node.index}"
    if node.is_boundary:
        label += " B"
    attr_dict = {
        "style": "filled",
        "shape": "circle",
        "label": label,
        "color": node.color.name,
        "fill_color": node.color.name,
    }
    return attr_dict


def edge_attr_fn(edge: GraphEdge):
    attr_dict = {
        "label": f"{edge.id}, {edge.index}",
    }
    return attr_dict


decoder = ConcatenatedDecoder([Color.red, Color.green, Color.yellow, Color.blue], construct_dual_graph())

graphviz_draw(decoder.dual_graph, node_attr_fn, edge_attr_fn, filename="mb_dualgraph.png", method="sfdp")
# https://stackoverflow.com/questions/14662618/is-there-a-3d-version-of-graphviz
exit()
# graphviz_draw(decoder.dual_graph, node_attr_fn, edge_attr_fn, filename="mb_dualgraph.vrml", method="sfdp", image_type="vrml", graph_attr={"dimen": "3"})
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow]), node_attr_fn, edge_attr_fn, filename="mb_restricted_gy.png", method="sfdp")
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow, Color.blue]), node_attr_fn, edge_attr_fn, filename="mb_restricted_gyb.png", method="sfdp")
graphviz_draw(decoder.mc3_graph([Color.green, Color.yellow], Color.blue), node_attr_fn, edge_attr_fn, filename="mb_monochrom_gy_b.png", method="sfdp")
graphviz_draw(decoder.mc4_graph([Color.green, Color.yellow], Color.blue, Color.red), node_attr_fn, edge_attr_fn, filename="mb_monochrom_gy_b_r.png", method="sfdp")
