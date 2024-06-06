import abc
import dataclasses
import itertools
from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import pymatching

from framework.layer import Syndrome
from framework.stabilizers import Stabilizer, Color
import rustworkx as rx
import matplotlib.pyplot as plt


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


@dataclass
class GraphObject(abc.ABC):
    # id given by us, is guaranteed to be the same between corresponding objects in different graphs and unique,
    # independently if the object is a node or an edge in the different graphs
    id: int = dataclasses.field(init=False)
    # index used by rustworkx, may be different between corresponding objects in different graphs
    index: int = dataclasses.field(init=False)

    def __eq__(self, other):
        if not isinstance(other, GraphObject):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, GraphObject):
            return NotImplemented
        return self.id < other.id


@dataclass
class GraphNode(GraphObject, abc.ABC):
    color: Color

    @property
    @abc.abstractmethod
    def is_boundary(self) -> bool:
        ...

    def get(self, attr, default):
        """Boilerplate code for pymatching."""
        if attr != "is_boundary":
            raise RuntimeError
        return self.is_boundary


@dataclass
class GraphEdge(GraphObject, abc.ABC):
    node1: GraphNode
    node2: GraphNode

    @property
    def is_edge_between_boundaries(self) -> bool:
        return self.node1.is_boundary and self.node2.is_boundary

    @property
    def node_ids(self) -> tuple[int, int]:
        """Easier access of node ids (first lower, then higher id)."""
        if self.node1.id < self.node2.id:
            return self.node1.id, self.node2.id
        else:
            return self.node2.id, self.node1.id

    def __contains__(self, item):
        """Boilerplate code for pymatching."""
        return False

    def get(self, attr, default):
        """Boilerplate code for pymatching."""
        if attr not in {"fault_ids", "weight", "error_probability"}:
            raise RuntimeError
        return default


# TODO add weight 0 to all edges linking only boundary nodes

@dataclass
class DualGraphNode(GraphNode):
    """Representation of one node in the dual lattice.

    A node corresponds to a stabilizer or a boundary of the primary lattice.
    """
    stabilizer: Optional[Stabilizer]
    # if this node corresponds to a boundary, we track here which qubits are at it.
    _adjacent_qubits: Optional[list[int]] = None

    def __post_init__(self):
        # perform sanity check: adjacent_qubits are given iff no stabilizer is given (i.e. this node is a boundary)
        if (self.stabilizer is None and self._adjacent_qubits is None
                or self.stabilizer is not None and self._adjacent_qubits is not None):
            raise ValueError
        if self.stabilizer is not None and self.stabilizer.color != self.color:
            raise ValueError
        if not self.color.is_monochrome:
            raise ValueError
        if self._adjacent_qubits:
            self._adjacent_qubits = sorted(self._adjacent_qubits)

    @property
    def is_boundary(self) -> bool:
        return self.stabilizer is None

    @property
    def qubits(self) -> list[int]:
        """Return the qubits which are adjacent to this stabilizer / boundary."""
        if self.is_boundary:
            return self._adjacent_qubits
        return self.stabilizer.qubits


@dataclass
class DualGraphEdge(GraphEdge):
    node1: DualGraphNode
    node2: DualGraphNode


@dataclass
class RestrictedGraphNode(GraphNode):
    """Representation of one node in a restricted lattice.

    A small wrapper around a DualGraphNode to get the indices right.
    """
    dg_node: DualGraphNode

    def __init__(self, dg_node: DualGraphNode) -> None:
        self.dg_node = dg_node

    @property
    def color(self):
        return self.dg_node.color

    @property
    def id(self):
        return self.dg_node.id

    @property
    def is_boundary(self):
        return self.dg_node.is_boundary

    @property
    def stabilizer(self):
        return self.dg_node.stabilizer

    @property
    def dg_nodes(self) -> list[DualGraphNode]:
        return [self.dg_node]

    @property
    def dg_indices(self) -> set[int]:
        return {self.dg_node.index}


@dataclass
class RestrictedGraphEdge(GraphEdge):
    node1: RestrictedGraphNode
    node2: RestrictedGraphNode
    dg_edge: DualGraphEdge

    @property
    def id(self):
        return self.dg_edge.id


@dataclass
class Mc3GraphNode(GraphNode):
    """Representation of one node of the monochromatic lattice with 3 colors.

    This builds upon a restricted lattice with 2 colors and adds a third color.

    A node corresponds to either:
    * a stabilizer (of the third color) (i.e. a node of the dual graph)
    * an edge of the restricted lattice (i.e. an edge of the dual graph)
    * a boundary (of the third color) (i.e. a node of the dual graph)
    """
    # edge of the restricted graph which corresponds to this node
    rg_edge: Optional[RestrictedGraphEdge] = None
    # node of the restricted graph which corresponds to this node
    rg_node: Optional[RestrictedGraphNode] = None

    @property
    def id(self) -> int:
        if self.rg_node:
            return self.rg_node.id
        elif self.rg_edge:
            return self.rg_edge.id
        raise NotImplementedError

    @property
    def is_boundary(self):
        if self.rg_node:
            return self.rg_node.is_boundary
        elif self.rg_edge:
            return self.rg_edge.is_edge_between_boundaries
        raise NotImplementedError

    @property
    def rg_nodes(self) -> list[RestrictedGraphNode]:
        if self.rg_node:
            return [self.rg_node]
        elif self.rg_edge:
            return [self.rg_edge.node1, self.rg_edge.node2]
        raise NotImplementedError

    @property
    def rg_indices(self) -> set[int]:
        if self.rg_node:
            return {self.rg_node.index}
        elif self.rg_edge:
            return {self.rg_edge.node1.index, self.rg_edge.node2.index}
        raise NotImplementedError

    @property
    def dg_nodes(self) -> list[DualGraphNode]:
        if self.rg_node:
            return self.rg_node.dg_nodes
        elif self.rg_edge:
            return [*self.rg_edge.node1.dg_nodes, *self.rg_edge.node2.dg_nodes]
        raise NotImplementedError

    @property
    def dg_indices(self) -> set[int]:
        if self.rg_node:
            return self.rg_node.dg_indices
        elif self.rg_edge:
            return self.rg_edge.node1.dg_indices | self.rg_edge.node2.dg_indices
        raise NotImplementedError


@dataclass
class Mc3GraphEdge(GraphEdge):
    id: int
    node1: Mc3GraphNode
    node2: Mc3GraphNode


@dataclass
class Mc4GraphNode(GraphNode):
    """Representation of one node of the monochromatic lattice with 4 colors.

    This builds upon the mc3 lattice and adds a fourth color.

    A node corresponds to either:
    * a stabilizer (of the fourth color) (i.e. a node of the dual graph)
    * an edge of the mc3 lattice (i.e. a face of the dual graph)
    * a boundary (of the fourth color) (i.e. a node of the dual graph)
    """
    # edge of the mc3 graph which corresponds to this node
    mc3_edge: Optional[Mc3GraphEdge] = None
    # node of the dual graph which corresponds to this node
    dg_node: Optional[DualGraphNode] = None

    @property
    def id(self) -> int:
        if self.dg_node:
            return self.dg_node.id
        if self.mc3_edge:
            return self.mc3_edge.id
        raise NotImplementedError

    @property
    def is_boundary(self):
        if self.dg_node:
            return self.dg_node.is_boundary
        if self.mc3_edge:
            return self.mc3_edge.is_edge_between_boundaries
        raise NotImplementedError

    @property
    def dg_nodes(self) -> list[DualGraphNode]:
        if self.dg_node:
            return [self.dg_node]
        elif self.mc3_edge:
            # Do manual deduplication of nodes. Does not use sets, since our custom classes are not hashable.
            return [k for k, _ in itertools.groupby(sorted((*self.mc3_edge.node1.dg_nodes, *self.mc3_edge.node2.dg_nodes)))]
        raise NotImplementedError

    @property
    def dg_indices(self) -> set[int]:
        if self.dg_node:
            return {self.dg_node.index}
        elif self.mc3_edge:
            return self.mc3_edge.node1.dg_indices | self.mc3_edge.node2.dg_indices
        raise NotImplementedError


@dataclass
class Mc4GraphEdge(GraphEdge):
    id: int
    node1: Mc4GraphNode
    node2: Mc4GraphNode

    @property
    def qubit(self) -> int:
        """Returns the qubit corresponding to this edge."""
        dg_nodes = [*self.node1.dg_nodes, *self.node2.dg_nodes]
        common_qubits = set(dg_nodes.pop().qubits)
        for node in dg_nodes:
            common_qubits &= set(node.qubits)
        if len(common_qubits) != 1:
            raise RuntimeError(common_qubits)
        return common_qubits.pop()


@dataclass
class ConcatenatedDecoder(Decoder):
    # all available colors of stabilizers of the code.
    # 2D Color Code: 3, 3D Color Code: 4.
    colors: list[Color]
    dual_graph: rx.PyGraph
    _restricted_graphs: dict[tuple[Color, ...], rx.PyGraph] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _restricted_matching_graphs: dict[tuple[Color, ...], pymatching.Matching] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _mc3_graphs: dict[tuple[Color, Color], dict[Color, rx.PyGraph]] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _mc3_matching_graphs: dict[tuple[Color, Color], dict[Color, pymatching.Matching]] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _mc4_graphs: dict[tuple[Color, Color], dict[Color, rx.PyGraph]] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _mc4_matching_graphs: dict[tuple[Color, Color], dict[Color, pymatching.Matching]] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _last_id: int = dataclasses.field(default=0, init=False)

    def __post_init__(self):
        # assign proper ids to dual graph nodes and edges
        for node in self.dual_graph.nodes():
            node.id = self.next_id
        for edge in self.dual_graph.edges():
            edge.id = self.next_id

    @property
    def next_id(self) -> int:
        """Generate the next unique id for a graph object."""
        self._last_id += 1
        return self._last_id

    def restricted_graph(self, colors: list[Color]) -> rx.PyGraph:
        """Construct the dual graph, restricted to nodes (and their edges) of the given colors."""
        if len(colors) not in {2, 3}:
            raise ValueError
        if self._restricted_graphs:
            return self._restricted_graphs[tuple(colors)]
        graphs = {}
        matching_graphs = {}
        for num_colors in [2, 3]:
            for restricted_colors in itertools.combinations(self.colors, num_colors):
                graph = self._construct_restricted_graph(restricted_colors)
                for cols in itertools.permutations(restricted_colors):
                    graphs[cols] = graph
                    matching_graphs[cols] = pymatching.Matching(graph)
        self._restricted_graphs = graphs
        self._restricted_matching_graphs = matching_graphs
        return self._restricted_graphs[tuple(colors)]

    def restricted_matching_graph(self, colors: list[Color]) -> pymatching.Matching:
        if not self._restricted_matching_graphs:
            self.restricted_graph(colors)
        return self._restricted_matching_graphs[tuple(colors)]

    def _construct_restricted_graph(self, colors: list[Color]) -> rx.PyGraph:
        graph = rx.PyGraph(multigraph=False)
        dual_graph = self.dual_graph
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

    def mc3_graph(self, restricted_colors: list[Color], monochromatic_color: Color) -> rx.PyGraph:
        """Construct the monochromatic graph with 3 colors based on the given colors.

        The nodes of this graph are {edges of restricted graph, stabilizers & boundary of third color}.
        The edges of this graph are the triangular faces of the restricted graph of all 3 colors.
        """
        if len(restricted_colors) != 2 or monochromatic_color in restricted_colors:
            raise ValueError
        rcolors = restricted_colors[0], restricted_colors[1]
        if self._mc3_graphs:
            return self._mc3_graphs[rcolors][monochromatic_color]
        graphs: dict[tuple[Color, ...], dict[Color, rx.PyGraph]] = {}
        matching_graphs = {}
        for r_colors in itertools.combinations(self.colors, 2):
            graphs[tuple(r_colors)] = {}
            graphs[tuple(r_colors[::-1])] = {}
            matching_graphs[tuple(r_colors)] = {}
            matching_graphs[tuple(r_colors[::-1])] = {}
            for m_color in set(self.colors) - set(r_colors):
                graph = self._construct_mc3_graph(r_colors, m_color)
                graphs[tuple(r_colors)][m_color] = graph
                graphs[tuple(r_colors[::-1])][m_color] = graph
                matching_graph = pymatching.Matching(graph)
                matching_graphs[tuple(r_colors)][m_color] = matching_graph
                matching_graphs[tuple(r_colors[::-1])][m_color] = matching_graph
        self._mc3_graphs = graphs
        self._mc3_matching_graphs = matching_graphs
        return self._mc3_graphs[rcolors][monochromatic_color]

    def mc3_matching_graph(self, restricted_colors: list[Color], monochromatic_color: Color) -> pymatching.Matching:
        if len(restricted_colors) != 2 or monochromatic_color in restricted_colors:
            raise ValueError
        rcolors = restricted_colors[0], restricted_colors[1]
        if not self._mc3_matching_graphs:
            self.mc3_graph(restricted_colors, monochromatic_color)
        return self._mc3_matching_graphs[rcolors][monochromatic_color]

    def _construct_mc3_graph(self, restricted_colors: list[Color], monochromatic_color: Color) -> rx.PyGraph:
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
            nodes.append(Mc3GraphNode(monochromatic_color, rg_edge=edges_by_id[edge.id]))
        # ... corresponding to nodes of the 3-colored restricted graph of color monochromatic_color
        for node in restricted_3_graph.nodes():
            if node.color == monochromatic_color:
                nodes.append(Mc3GraphNode(monochromatic_color, rg_node=node))
        graph.add_nodes_from(nodes)
        for index in graph.node_indices():
            graph[index].index = index

        rg_node_to_mc_node: dict[tuple[int, ...], Mc3GraphNode] = {}
        for node in nodes:
            if tuple(node.rg_indices) in rg_node_to_mc_node:
                raise RuntimeError
            for permutation in itertools.permutations(node.rg_indices):
                rg_node_to_mc_node[tuple(permutation)] = node

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
            if (cycle[0], ) in rg_node_to_mc_node:
                node1 = rg_node_to_mc_node[(cycle[0], )]
                node2 = rg_node_to_mc_node[(cycle[1], cycle[2])]
            elif (cycle[1], ) in rg_node_to_mc_node:
                node1 = rg_node_to_mc_node[(cycle[1], )]
                node2 = rg_node_to_mc_node[(cycle[0], cycle[2])]
            elif (cycle[2], ) in rg_node_to_mc_node:
                node1 = rg_node_to_mc_node[(cycle[2], )]
                node2 = rg_node_to_mc_node[(cycle[0], cycle[1])]
            else:
                raise RuntimeError
            # the edge which belongs to three boundary nodes belongs to no qubit
            # TODO add weight 0
            # if node1.is_boundary and node2.is_boundary:
            #     continue
            graph.add_edge(node1.index, node2.index, Mc3GraphEdge(self.next_id, node1, node2))

        for index in graph.edge_indices():
            graph.edge_index_map()[index][2].index = index

        return graph

    def mc4_graph(self, restricted_colors: list[Color], monochromatic_3_color: Color, monochromatic_4_color: Color) -> rx.PyGraph:
        """Construct the monochromatic graph with 4 colors based on the given colors.

        The nodes of this graph are {edges of mc3-graph, stabilizers & boundary of fourth color}.
        The edges of this graph are the tetrahedral volumes of the dual graph of all 4 colors.
        """
        if (len(restricted_colors) != 2
                or monochromatic_3_color in restricted_colors or monochromatic_4_color in restricted_colors
                or monochromatic_3_color == monochromatic_4_color):
            raise ValueError
        rcolors = restricted_colors[0], restricted_colors[1]
        if self._mc4_graphs:
            return self._mc4_graphs[rcolors][monochromatic_3_color]
        graphs: dict[tuple[Color, Color], dict[Color, rx.PyGraph]] = {}
        matching_graphs = {}
        for r_colors in itertools.combinations(self.colors, 2):
            graphs[r_colors] = {}
            graphs[r_colors[::-1]] = {}
            matching_graphs[r_colors] = {}
            matching_graphs[r_colors[::-1]] = {}
            for m_3_color in set(self.colors) - set(r_colors):
                m_4_color = (set(self.colors) - set(r_colors) - {m_3_color}).pop()
                graph = self._construct_mc4_graph(r_colors, m_3_color, m_4_color)
                graphs[r_colors][m_3_color] = graph
                graphs[r_colors[::-1]][m_3_color] = graph
                matching_graph = pymatching.Matching(graph)
                matching_graphs[r_colors][m_3_color] = matching_graph
                matching_graphs[r_colors[::-1]][m_3_color] = matching_graph
        self._mc4_graphs = graphs
        self._mc4_matching_graphs = matching_graphs
        return self._mc4_graphs[rcolors][monochromatic_3_color]

    def mc4_matching_graph(self, restricted_colors: list[Color], monochromatic_3_color: Color, monochromatic_4_color: Color) -> pymatching.Matching:
        if (len(restricted_colors) != 2
                or monochromatic_3_color in restricted_colors or monochromatic_4_color in restricted_colors
                or monochromatic_3_color == monochromatic_4_color):
            raise ValueError
        rcolors = restricted_colors[0], restricted_colors[1]
        if not self._mc4_matching_graphs:
            self.mc4_graph(restricted_colors, monochromatic_3_color, monochromatic_4_color)
        return self._mc4_matching_graphs[rcolors][monochromatic_3_color]

    def _construct_mc4_graph(self, restricted_colors: list[Color], monochromatic_3_color: Color, monochromatic_4_color: Color) -> rx.PyGraph:
        graph = rx.PyGraph(multigraph=False)
        mc3_graph = self.mc3_graph(restricted_colors, monochromatic_3_color)
        dual_graph = self.dual_graph
        # rustworkx guarantees that the undirected and directed graph share the same indexes for the same objects.
        directed_dual_graph = dual_graph.to_directed()

        # construct the nodes
        nodes = []
        # ... corresponding to edges of the mc3 graph
        for edge in mc3_graph.edges():
            nodes.append(Mc4GraphNode(monochromatic_4_color, mc3_edge=edge))
        # ... corresponding to nodes of the dual graph of color monochromatic_4_color
        for node in dual_graph.nodes():
            if node.color == monochromatic_4_color:
                nodes.append(Mc4GraphNode(monochromatic_4_color, dg_node=node))
        graph.add_nodes_from(nodes)
        for index in graph.node_indices():
            graph[index].index = index

        dg_node_to_mc4_node: dict[tuple[int, ...], Mc4GraphNode] = {}
        for node in nodes:
            if tuple(node.dg_indices) in dg_node_to_mc4_node:
                raise RuntimeError
            for permutation in itertools.permutations(node.dg_indices):
                dg_node_to_mc4_node[tuple(permutation)] = node

        # construct the edges
        # first, we find each 3-cycle (corresponding to a triangle face) of the lattice
        cycles = {tuple(sorted(indices)) for indices in rx.simple_cycles(directed_dual_graph) if len(indices) == 3}
        # then, we find for each 3-cycle all nodes which are connected with all of them, forming a tetrahedron
        tetrahedrons = set()
        for cycle in cycles:
            common_neighbours = set(dual_graph.neighbors(cycle[0])) & set(dual_graph.neighbors(cycle[1])) & set(dual_graph.neighbors(cycle[2]))
            for neighbour in common_neighbours:
                tetrahedrons.add(tuple(sorted([*cycle, neighbour])))
        # TODO adjust this to the length of qubits of one side of the tetrahedron in general + boundary-boundary-boundary
        if len(tetrahedrons) != 16:
            raise RuntimeError(len(tetrahedrons))
        for tetrahedron in tetrahedrons:
            # one of the nodes of the tetrahedron belongs to a node
            # three of the nodes of the tetrahedron belong to an edge of the mc3 lattice
            if (tetrahedron[0], ) in dg_node_to_mc4_node:
                node1 = dg_node_to_mc4_node[(tetrahedron[0],)]
                node2 = dg_node_to_mc4_node[(tetrahedron[1], tetrahedron[2], tetrahedron[3])]
            elif (tetrahedron[1], ) in dg_node_to_mc4_node:
                node1 = dg_node_to_mc4_node[(tetrahedron[1],)]
                node2 = dg_node_to_mc4_node[(tetrahedron[0], tetrahedron[2], tetrahedron[3])]
            elif (tetrahedron[2], ) in dg_node_to_mc4_node:
                node1 = dg_node_to_mc4_node[(tetrahedron[2],)]
                node2 = dg_node_to_mc4_node[(tetrahedron[0], tetrahedron[1], tetrahedron[3])]
            elif (tetrahedron[3], ) in dg_node_to_mc4_node:
                node1 = dg_node_to_mc4_node[(tetrahedron[3],)]
                node2 = dg_node_to_mc4_node[(tetrahedron[0], tetrahedron[1], tetrahedron[2])]
            else:
                raise RuntimeError
            # the edge which belongs to four boundary nodes belongs to no qubit
            # TODO add weight 0 to this edge
            # if node1.is_boundary and node2.is_boundary:
            #     continue
            graph.add_edge(node1.index, node2.index, Mc4GraphEdge(self.next_id, node1, node2))

        for index in graph.edge_indices():
            graph.edge_index_map()[index][2].index = index

        return graph

    def _matching2nodes(self, edges: list[tuple[int, int]], edge_graph: rx.PyGraph, node_graph: rx.PyGraph) -> list[GraphNode]:
        """Take the edges (pairs of indices) of edge_graph and return the corresponding nodes of node_graph."""
        edges_ = self._matching2edges(edges, edge_graph)
        ids2nodes = {node.id: node for node in node_graph.nodes()}
        return [ids2nodes[edge.id] for edge in edges_]

    @staticmethod
    def _matching2edges(edges: list[tuple[int, int]], edge_graph: rx.PyGraph) -> list[GraphEdge]:
        """Take the edges (pairs of indices) of edge_graph and return the corresponding edges of edge_graph."""
        # TODO should be worthwhile to cache this whole function, i.e. compute this mapping of edges -> GraphEdges
        #  for all relevant cases and save it as a dict
        indices2ids = {node.index: node.id for node in edge_graph.nodes()}
        ids2edges = {edge.node_ids: edge for edge in edge_graph.edges()}
        boundary_nodes = [node for node in edge_graph.nodes() if node.is_boundary]
        ret = []
        for index1, index2 in edges:
            # special case: index1 was matched against a boundary.
            if index2 == -1:
                # find a boundary node which shares an edge with the node with index1
                # TODO on restricted lattices, the matched boundary should be unique. On monochromatic, it should not
                #  matter which one we choose (if there are multiple).
                for boundary in boundary_nodes:
                    if edge_graph.has_edge(index1, boundary.index):
                        index2 = boundary.index
                        break
            id1 = indices2ids[index1]
            id2 = indices2ids[index2]
            # fix sorting of id1 and id2
            if id1 > id2:
                id1, id2 = id2, id1
            # get the edge of edge_graph
            ret.append(ids2edges[id1, id2])
        return ret

    def decode(self, syndrome: Syndrome) -> list[int]:
        possible_corrections: list[list[int]] = []

        for r_colors in itertools.combinations(self.colors, 2):
            # create restricted graph of this colors
            r_graph = self.restricted_graph(r_colors)

            # construct the syndrome input for the given graph
            r_syndrome = []
            for node in sorted(r_graph.nodes(), key=lambda x: x.index):
                node: RestrictedGraphNode
                if node.is_boundary:
                    r_syndrome.append(False)
                else:
                    r_syndrome.append(syndrome[node.stabilizer].ancilla)

            # perform the first matching
            r_matching_graph = self.restricted_matching_graph(r_colors)
            r_matching = r_matching_graph.decode_to_edges_array(r_syndrome)

            # plt.figure()
            # r_matching.draw()
            # plt.savefig(fname="pymatching.png")
            # pprint(r_matching.edges())

            for lifting_colors in itertools.permutations(set(self.colors) - set(r_colors), 2):
                mc3_color, mc4_color = lifting_colors
                mc3_graph = self.mc3_graph(r_colors, mc3_color)
                mc4_graph = self.mc4_graph(r_colors, mc3_color, mc4_color)

                # construct the syndrome input from the previous matching and mc3_color-stabilizers.
                mc3_nodes = self._matching2nodes(r_matching, r_graph, mc3_graph)
                mc3_syndrome = []
                for node in sorted(mc3_graph.nodes(), key=lambda x: x.index):
                    node: Mc3GraphNode
                    if node.is_boundary:
                        if node in mc3_nodes:
                            raise RuntimeError
                        mc3_syndrome.append(False)
                    elif node in mc3_nodes:
                        mc3_syndrome.append(True)
                    elif node.rg_node is not None and node.rg_node.stabilizer.color == mc3_color:
                        mc3_syndrome.append(syndrome[node.rg_node.stabilizer].ancilla)
                    else:
                        mc3_syndrome.append(False)

                # perform the second matching
                mc3_matching_graph = self.mc3_matching_graph(r_colors, mc3_color)
                mc3_matching = mc3_matching_graph.decode_to_edges_array(mc3_syndrome)

                # construct the syndrome input from the previous matching and mc4_color-stabilizers.
                mc4_nodes = self._matching2nodes(mc3_matching, mc3_graph, mc4_graph)
                mc4_syndrome = []
                for node in sorted(mc4_graph.nodes(), key=lambda x: x.index):
                    node: Mc4GraphNode
                    if node.is_boundary:
                        if node in mc4_nodes:
                            raise RuntimeError
                        mc4_syndrome.append(False)
                    elif node in mc4_nodes:
                        mc4_syndrome.append(True)
                    elif node.dg_node is not None and node.dg_node.stabilizer.color == mc4_color:
                        mc4_syndrome.append(syndrome[node.dg_node.stabilizer].ancilla)
                    else:
                        mc4_syndrome.append(False)

                # perform the third matching
                mc4_matching_graph = self.mc4_matching_graph(r_colors, mc3_color, mc4_color)
                mc4_matching = mc4_matching_graph.decode_to_edges_array(mc4_syndrome)
                mc4_edges: list[Mc4GraphEdge] = self._matching2edges(mc4_matching, mc4_graph)

                possible_corrections.append(sorted([edge.qubit for edge in mc4_edges]))
                # print([r_colors, mc3_color, mc4_color, possible_corrections[-1]])

        # return the lowest-weight correction
        return min(possible_corrections, key=len)
