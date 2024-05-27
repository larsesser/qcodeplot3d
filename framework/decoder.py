import abc
import dataclasses
import enum
import itertools
from dataclasses import dataclass
from copy import deepcopy
from pprint import pprint
# from framework.layer import Syndrome
Syndrome = None
from typing import Optional
import rustworkx as rx
from rustworkx.visualization import graphviz_draw, mpl_draw


OBJECT_ID: int = 1


@dataclass
class Decoder(abc.ABC):

    @abc.abstractmethod
    def decode(self, syndrome: Syndrome) -> list[int]:
        """Determine the qubits on which a correction shall be applied from a given syndrome."""


# Mapping of syndromes (detector events) to qubits of the correction.
LookupTable = dict[tuple[bool, ...], list[int]]


@dataclass
class LookupTableDecoder(Decoder):
    lookup_table: LookupTable

    def decode(self, syndrome: Syndrome) -> list[int]:
        """Determine the qubits on which a correction shall be applied from a given syndrome.

        Use a (non-scalable) lookup table for decoding.
        """
        return self.lookup_table.get(syndrome.value, [])


class Color(enum.IntEnum):
    red = 1
    blue = 2
    green = 3
    yellow = 4


@dataclass
class GraphNode:
    color: Color
    # id given by us, is guaranteed to be the same between corresponding objects in different graphs and unique,
    # independently if the object is a node or an edge in the different graphs
    id: int

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id


@dataclass
class DualGraphNode(GraphNode):
    """Representation of one node in a dual (or restricted) lattice.

    A node corresponds to a stabilizer or a boundary of the primary lattice.
    """
    is_boundary: bool = False
    # index used by rustworkx, may be different between corresponding objects in different graphs
    index: int = dataclasses.field(default=int, init=False)
    # TODO: stabilizer

    @property
    def is_stabilizer(self):
        return not self.is_boundary


@dataclass
class DualGraphEdge:
    node1: DualGraphNode
    node2: DualGraphNode
    # id given by us, is guaranteed to be the same between corresponding objects in different graphs and unique,
    # independently if the object is a node or an edge in the different graphs
    id: int
    index: int = dataclasses.field(default=int, init=False)

    @property
    def is_edge_between_boundaries(self):
        return self.node1.is_boundary and self.node2.is_boundary


@dataclass
class Mc3GraphNode(GraphNode):
    """Representation of one node of the monochromatic lattice with 3 colors.

    This builds upon a restricted lattice with 2 colors and adds a third color.

    A node corresponds to either:
    * a stabilizer (of the third color) (i.e. a node of the dual graph)
    * an edge of the restricted lattice (i.e. an edge of the dual graph)
    * a boundary (of the third color) (i.e. a node of the dual graph)
    """
    # edge of the dual graph which corresponds to this node
    dg_edge: Optional[DualGraphEdge] = None
    # node of the dual graph which corresponds to this node
    dg_node: Optional[DualGraphNode] = None
    index: int = dataclasses.field(default=int, init=False)

    @property
    def is_edge(self):
        return self.dg_edge is not None

    @property
    def is_boundary(self):
        if self.dg_edge is None:
            return False
        return self.dg_node.is_boundary


@dataclass
class ConcatenatedDecoder(Decoder):
    # all available colors of stabilizers of the code.
    # 2D Color Code: 3, 3D Color Code: 4.
    colors: list[Color]
    _dual_graph: rx.PyGraph = dataclasses.field(default=None, init=False)
    _restricted_graphs: dict[tuple[Color, ...], rx.PyGraph] = dataclasses.field(default=None, init=False)
    _mc3_graphs: dict[tuple[Color, ...], dict[Color, rx.PyGraph]] = dataclasses.field(default=None, init=False)

    def primary_graph(self):
        ...

    def dual_graph(self) -> rx.PyGraph:
        if self._dual_graph:
            return self._dual_graph
        global OBJECT_ID
        graph = rx.PyGraph(multigraph=False)
        # two nodes per Color
        for color in self.colors:
            graph.add_nodes_from([DualGraphNode(color, OBJECT_ID), DualGraphNode(color, OBJECT_ID+1, is_boundary=True)])
            OBJECT_ID += 2
        for index in graph.node_indices():
            graph[index].index = index
        # insert edges between the nodes
        for node1, node2 in itertools.combinations(graph.nodes(), 2):
            if node1.color == node2.color or graph.has_edge(node1.index, node2.index):
                continue
            graph.add_edge(node1.index, node2.index, DualGraphEdge(node1, node2, OBJECT_ID))
            OBJECT_ID += 1
        for index in graph.edge_indices():
            graph.edge_index_map()[index][2].index = index
        self._dual_graph = graph
        return graph

    def restricted_graph(self, colors: list[Color]) -> rx.PyGraph:
        """Construct the dual graph, restricted to nodes (and their edges) of the given colors."""
        if self._restricted_graphs:
            return self._restricted_graphs[tuple(colors)]
        dual_graph = self.dual_graph()
        restricted_graphs = {}
        for num_colors in [2, 3]:
            for restricted_colors in itertools.combinations(self.colors, num_colors):
                # restricted_graph = deepcopy(dual_graph)
                # nodes = restricted_graph.nodes()
                # for node in nodes:
                #     if node.color in restricted_colors:
                #         continue
                #     # remove edges and nodes of other colors
                #     for _, _, edge in restricted_graph.in_edges(node.index):
                #         restricted_graph.remove_edge_from_index(edge.index)
                #     restricted_graph.remove_node(node.index)
                #     for cols in itertools.permutations(restricted_colors):
                #         restricted_graphs[cols] = restricted_graph
                restricted_graph = rx.PyGraph(multigraph=False)
                for node in dual_graph.nodes():
                    if node.color in restricted_colors:
                        restricted_graph.add_node(deepcopy(node))
                for index in restricted_graph.node_indices():
                    restricted_graph[index].index = index
                nodes_by_id = {node.id: node for node in restricted_graph.nodes()}
                for orig_edge in dual_graph.edges():
                    if orig_edge.node1.color in restricted_colors and orig_edge.node2.color in restricted_colors:
                        edge = deepcopy(orig_edge)
                        edge.node1 = nodes_by_id[edge.node1.id]
                        edge.node2 = nodes_by_id[edge.node2.id]
                        restricted_graph.add_edge(edge.node1.index, edge.node2.index, edge)
                for index in restricted_graph.edge_indices():
                    restricted_graph.edge_index_map()[index][2].index = index
                for cols in itertools.permutations(restricted_colors):
                    restricted_graphs[cols] = restricted_graph
        return restricted_graphs[tuple(colors)]

    def mc3_graph(self, restricted_colors: list[Color], monochromatic_color: Color) -> rx.PyGraph:
        """Construct the monochromatic graph with 3 colors based on the given colors.

        The nodes of this graph are {edges of restricted graph [except edges between two boundaries], stabilizers + boundary of third color}.
        The edges of this graph are the triangular faces of the restricted graph of all 3 colors."""
        if len(restricted_colors) != 2:
            raise ValueError
        if self._mc3_graphs:
            return self._mc3_graphs[tuple(restricted_colors)][monochromatic_color]
        graph = rx.PyGraph(multigraph=False)

        # construct the nodes
        nodes = []
        # ... corresponding to edges of the 2-colored restricted graph
        for edge in self.restricted_graph(restricted_colors).edges():
            if edge.is_edge_between_boundaries:
                continue
            nodes.append(Mc3GraphNode(monochromatic_color, id=edge.id, dg_edge=edge))
        # ... corresponding to nodes of the 3-colored restricted graph of color monochromatic_color
        for node in self.restricted_graph([*restricted_colors, monochromatic_color]).nodes():
            if node.color == monochromatic_color:
                nodes.append(Mc3GraphNode(monochromatic_color, id=node.id, dg_node=node))
        graph.add_nodes_from(nodes)
        for index in graph.node_indices():
            graph[index].index = index
        pprint(nodes)

        # construct the edges
        restricted_graph = self.restricted_graph([*restricted_colors, monochromatic_color]).to_directed()
        graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow, Color.blue]), node_attr_fn, edge_attr_fn, filename="restricted_gyb2.png", method="sfdp")
        cycles = {tuple(sorted(indices)) for indices in rx.simple_cycles(restricted_graph) if len(indices) == 3}
        rg_node_to_mc_node = {}
        for node in nodes:
            if node.dg_node and node.dg_edge:
                raise RuntimeError
            if node.dg_node:
                if node.dg_node.index in rg_node_to_mc_node:
                    raise RuntimeError
                rg_node_to_mc_node[node.dg_node.index] = node
            elif node.dg_edge:
                if node.dg_edge.node1.index or node.dg_edge.node2.index in rg_node_to_mc_node:
                    raise RuntimeError
                rg_node_to_mc_node[node.dg_edge.node1.index] = node
                rg_node_to_mc_node[node.dg_edge.node2.index] = node
            else:
                raise RuntimeError
        print(cycles)
        print(list(map(lambda x: [rg_node_to_mc_node[i].id for i in x], cycles)))
        # TODO adjust this to the length of qubits of one side of the tetrahedron in general
        if len(cycles) != 7:
            raise RuntimeError
        print(cycles)
        for cycle in cycles:
            # each face (== cycle) of the graph is a triangle
            if len(cycle) != 3:
                raise RuntimeError(cycle)
            # two of the nodes of the triangle belong to an edge which is a node in the mc lattice,
            # one of the nodes of the triangle belongs to a node
            node1, node2, node3 = rg_node_to_mc_node[cycle[0]], rg_node_to_mc_node[cycle[1]], rg_node_to_mc_node[cycle[2]]
            if node1 == node2:
                graph.add_edge(node1.index, node3.index, None)
            elif node1 == node3:
                graph.add_edge(node1.index, node2.index, None)
            elif node2 == node3:
                graph.add_edge(node1.index, node2.index, None)
            else:
                raise RuntimeError(f"\n{node1}\n{node2}\n{node3}")

        return graph

    def decode(self, syndrome: Syndrome) -> list[int]:
        possible_corrections: list[list[int]] = []

        for restriction_colors in itertools.combinations(self.colors, 2):
            # create restricted graph of this colors
            monochromatic_colors = set(self.colors) - set(restriction_colors)
            for lifting_colors in itertools.permutations(monochromatic_colors, 2):
                for monochromatic_color in lifting_colors:
                    # create monochromatic graph of this color (with restriction_colors and all previous monochromatic_color)
                    possible_corrections.append([0])

        # return the lowest-weight correction
        return min(possible_corrections, key=len)


def node_attr_fn(node: DualGraphNode):
    label = f"{node.index}, {node.id}"
    if node.is_boundary:
        label += "B"
    attr_dict = {
        "style": "filled",
        "shape": "circle",
        "label": label,
        "color": node.color.name,
        "fill_color": node.color.name,
    }
    return attr_dict


def edge_attr_fn(edge: DualGraphEdge):
    attr_dict = {
        "label": f"{edge.index}, {edge.id}"
    }
    return attr_dict


decoder = ConcatenatedDecoder([Color.red, Color.green, Color.yellow, Color.blue])
graphviz_draw(decoder.dual_graph(), node_attr_fn, edge_attr_fn, filename="dualgraph.png", method="sfdp")
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow]), node_attr_fn, edge_attr_fn, filename="restricted_gy.png", method="sfdp")
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow, Color.blue]), node_attr_fn, edge_attr_fn, filename="restricted_gyb.png", method="sfdp")
decoder.mc3_graph([Color.green, Color.yellow], Color.blue)
