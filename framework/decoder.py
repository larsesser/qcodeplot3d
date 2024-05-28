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
    # index used by rustworkx, may be different between corresponding objects in different graphs
    index: int = dataclasses.field(default=int, init=False)

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id


@dataclass
class GraphEdge:
    node1: GraphNode
    node2: GraphNode
    # id given by us, is guaranteed to be the same between corresponding objects in different graphs and unique,
    # independently if the object is a node or an edge in the different graphs
    id: int
    # index used by rustworkx, may be different between corresponding objects in different graphs
    index: int = dataclasses.field(default=int, init=False)


@dataclass
class DualGraphNode(GraphNode):
    """Representation of one node in a dual (or restricted) lattice.

    A node corresponds to a stabilizer or a boundary of the primary lattice.
    """
    is_boundary: bool = False
    # TODO: stabilizer

    @property
    def is_stabilizer(self):
        return not self.is_boundary


@dataclass
class DualGraphEdge(GraphEdge):
    node1: DualGraphNode
    node2: DualGraphNode

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

    @property
    def is_edge(self):
        return self.dg_edge is not None

    @property
    def is_boundary(self):
        if self.dg_node:
            return self.dg_node.is_boundary
        if self.dg_edge:
            return self.dg_edge.is_edge_between_boundaries
        raise NotImplementedError


@dataclass
class Mc3GraphEdge(GraphEdge):
    node1: Mc3GraphNode
    node2: Mc3GraphNode


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
        graphs = {}
        for num_colors in [2, 3]:
            for restricted_colors in itertools.combinations(self.colors, num_colors):
                graph = self._construct_restricted_graph(restricted_colors)
                for cols in itertools.permutations(restricted_colors):
                    graphs[cols] = graph
        self._restricted_graphs = graphs
        return self._restricted_graphs[tuple(colors)]

    def _construct_restricted_graph(self, colors: list[Color]) -> rx.PyGraph:
        graph = rx.PyGraph(multigraph=False)
        dual_graph = self.dual_graph()
        for node in dual_graph.nodes():
            if node.color in colors:
                graph.add_node(deepcopy(node))
        for index in graph.node_indices():
            graph[index].index = index
        nodes_by_id = {node.id: node for node in graph.nodes()}
        # take care to get the properties of the nodes associated to an edge right
        for orig_edge in dual_graph.edges():
            if orig_edge.node1.color in colors and orig_edge.node2.color in colors:
                edge = deepcopy(orig_edge)
                edge.node1 = nodes_by_id[edge.node1.id]
                edge.node2 = nodes_by_id[edge.node2.id]
                graph.add_edge(edge.node1.index, edge.node2.index, edge)
        for index in graph.edge_indices():
            graph.edge_index_map()[index][2].index = index
        return graph

    def mc3_graph(self, restricted_colors: list[Color], monochromatic_color: Color) -> rx.PyGraph:
        """Construct the monochromatic graph with 3 colors based on the given colors.

        The nodes of this graph are {edges of restricted graph, stabilizers & boundary of third color}.
        The edges of this graph are the triangular faces of the restricted graph of all 3 colors.
        """
        if len(restricted_colors) != 2 or monochromatic_color in restricted_colors:
            raise ValueError
        if self._mc3_graphs:
            return self._mc3_graphs[tuple(restricted_colors)][monochromatic_color]
        graphs: dict[tuple[Color, ...], dict[Color, rx.PyGraph]] = {}
        for r_colors in itertools.combinations(self.colors, 2):
            graphs[tuple(r_colors)] = {}
            graphs[tuple(r_colors[::-1])] = {}
            for m_color in set(self.colors) - set(r_colors):
                graph = self._construct_mc3_graph(r_colors, m_color)
                graphs[tuple(r_colors)][m_color] = graph
                graphs[tuple(r_colors[::-1])][m_color] = graph
        self._mc3_graphs = graphs
        return self._mc3_graphs[tuple(restricted_colors)][monochromatic_color]

    def _construct_mc3_graph(self, restricted_colors: list[Color], monochromatic_color: Color) -> rx.PyGraph:
        global OBJECT_ID
        graph = rx.PyGraph(multigraph=False)
        restricted_2_graph = self.restricted_graph(restricted_colors)
        restricted_3_graph = self.restricted_graph([*restricted_colors, monochromatic_color])
        # since we didn't remove any objects to construct the restricted graph, rustworkx guarantees that the
        #  undirected and directed graph share the same indexes for the same objects.
        directed_restricted_3_graph = restricted_3_graph.to_directed()

        # construct the nodes
        nodes = []
        # take care: since we construct the edges via nodes of the 3-restricted graph, we use the edges from the
        #  3-restricted graph to ensure the indices are identical.
        edges_by_id = {edge.id: edge for edge in restricted_3_graph.edges()}
        # ... corresponding to edges of the 2-colored restricted graph
        for edge in restricted_2_graph.edges():
            nodes.append(Mc3GraphNode(monochromatic_color, id=edge.id, dg_edge=edges_by_id[edge.id]))
        # ... corresponding to nodes of the 3-colored restricted graph of color monochromatic_color
        for node in restricted_3_graph.nodes():
            if node.color == monochromatic_color:
                nodes.append(Mc3GraphNode(monochromatic_color, id=node.id, dg_node=node))
        graph.add_nodes_from(nodes)
        for index in graph.node_indices():
            graph[index].index = index

        rg_node_to_mc_node: dict[int | tuple[int, int], Mc3GraphNode] = {}
        for node in nodes:
            if node.dg_node and node.dg_edge:
                raise RuntimeError
            if node.dg_node:
                if node.dg_node.index in rg_node_to_mc_node:
                    raise RuntimeError
                rg_node_to_mc_node[node.dg_node.index] = node
            elif node.dg_edge:
                if (node.dg_edge.node1.index, node.dg_edge.node2.index) in rg_node_to_mc_node:
                    raise RuntimeError
                rg_node_to_mc_node[(node.dg_edge.node1.index, node.dg_edge.node2.index)] = node
                rg_node_to_mc_node[(node.dg_edge.node2.index, node.dg_edge.node1.index)] = node
            else:
                raise RuntimeError

        # graphviz_draw(restricted_3_graph, node_attr_fn, edge_attr_fn, filename="restricted_gyb2.png", method="sfdp")
        # graphviz_draw(directed_restricted_3_graph, node_attr_fn, edge_attr_fn, filename="restricted_gyb2_direct.png", method="sfdp")

        # construct the edges
        cycles = {tuple(sorted(indices)) for indices in rx.simple_cycles(directed_restricted_3_graph) if len(indices) == 3}
        # TODO adjust this to the length of qubits of one side of the tetrahedron in general + boundary-boundary-boundary
        if len(cycles) != 8:
            raise RuntimeError
        for cycle in cycles:
            # each face (== cycle) of the graph is a triangle
            if len(cycle) != 3:
                raise RuntimeError(cycle)
            # one of the nodes of the triangle belongs to a node
            # two of the nodes of the triangle belong to an edge which is a node in the mc lattice,
            if cycle[0] in rg_node_to_mc_node:
                node1 = rg_node_to_mc_node[cycle[0]]
                node2 = rg_node_to_mc_node[(cycle[1], cycle[2])]
            elif cycle[1] in rg_node_to_mc_node:
                node1 = rg_node_to_mc_node[cycle[1]]
                node2 = rg_node_to_mc_node[(cycle[0], cycle[2])]
            elif cycle[2] in rg_node_to_mc_node:
                node1 = rg_node_to_mc_node[cycle[2]]
                node2 = rg_node_to_mc_node[(cycle[0], cycle[1])]
            else:
                raise RuntimeError
            # the edge which belongs to three boundary nodes belongs to no qubit
            if node1.dg_node.is_boundary and node2.dg_edge.node1.is_boundary and node2.dg_edge.node2.is_boundary:
                continue
            graph.add_edge(node1.index, node2.index, Mc3GraphEdge(node1, node2, OBJECT_ID))
            OBJECT_ID += 1

        for index in graph.edge_indices():
            graph.edge_index_map()[index][2].index = index

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
graphviz_draw(decoder.mc3_graph([Color.green, Color.yellow], Color.blue), node_attr_fn, edge_attr_fn, filename="monochrom_gy_b.png", method="sfdp")
