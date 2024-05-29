import abc
import dataclasses
import enum
import itertools
from dataclasses import dataclass
from copy import deepcopy
from pprint import pprint
from typing import Optional
from framework.layer import Syndrome
from framework.stabilizers import Stabilizer
import rustworkx as rx
# import pymatching as pm


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


class Color(enum.Enum):
    red = 1
    blue = 2
    green = 3
    yellow = 4


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


@dataclass
class GraphEdge(GraphObject, abc.ABC):
    node1: GraphNode
    node2: GraphNode

    @property
    def is_edge_between_boundaries(self) -> bool:
        return self.node1.is_boundary and self.node2.is_boundary


# TODO add weight 0 to all edges linking only boundary nodes

@dataclass
class DualGraphNode(GraphNode):
    """Representation of one node in the dual lattice.

    A node corresponds to a stabilizer or a boundary of the primary lattice.
    """
    id: int
    stabilizer: Optional[Stabilizer]
    # if this node corresponds to a boundary, we track here which qubits are at it.
    _adjacent_qubits: Optional[list[int]] = None

    def __post_init__(self):
        # perform sanity check: adjacent_qubits are given iff no stabilizer is given (i.e. this node is a boundary)
        if (self.stabilizer is None and self._adjacent_qubits is None
                or self.stabilizer is not None and self._adjacent_qubits is not None):
            raise ValueError

    @property
    def is_boundary(self) -> bool:
        return self.stabilizer is None

    @property
    def qubits(self) -> list[int]:
        """Return the qubits which are adjacent to this stabilizer / boundary."""
        if self.is_boundary:
            return sorted(self._adjacent_qubits)
        return self.stabilizer.qubits


@dataclass
class DualGraphEdge(GraphEdge):
    id: int
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
    _dual_graph: rx.PyGraph = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _restricted_graphs: dict[tuple[Color, ...], rx.PyGraph] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _mc3_graphs: dict[tuple[Color, Color], dict[Color, rx.PyGraph]] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]
    _mc4_graphs: dict[tuple[Color, Color], dict[Color, rx.PyGraph]] = dataclasses.field(default=None, init=False)  # type: ignore[arg-type]

    def primary_graph(self):
        ...

    def dual_graph(self) -> rx.PyGraph:
        if self._dual_graph:
            return self._dual_graph
        global OBJECT_ID
        graph = rx.PyGraph(multigraph=False)

        def t2p(l_in: list[str]) -> list[int]:
            trans = {"A": 15, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "M": 11,
                     "N": 12, "P": 13, "Q": 14}
            return [trans[e] for e in l_in]

        # hard coded construction of [[15,1,3]] 3D Tetrahedral Color Code
        nodes = [
            DualGraphNode(OBJECT_ID+0, Color.red, Stabilizer(15, x_positions=t2p(["A", "J", "K", "H", "M", "Q", "I", "P"]))),
            DualGraphNode(OBJECT_ID+1, Color.yellow, Stabilizer(15, x_positions=t2p(["B", "F", "N", "G", "J", "K", "M", "Q"]))),
            DualGraphNode(OBJECT_ID+2, Color.blue, Stabilizer(15, x_positions=t2p(["F", "C", "E", "N", "M", "I", "P", "Q"]))),
            DualGraphNode(OBJECT_ID+3, Color.green, Stabilizer(15, x_positions=t2p(["D", "G", "N", "E", "H", "K", "Q", "P"]))),
            DualGraphNode(OBJECT_ID+4, Color.red, None, t2p(["B", "F", "C", "E", "D", "G", "N"])),
            DualGraphNode(OBJECT_ID+5, Color.yellow, None, t2p(["A", "I", "C", "E", "D", "H", "P"])),
            DualGraphNode(OBJECT_ID+6, Color.blue, None, t2p(["A", "J", "B", "G", "D", "H", "K"])),
            DualGraphNode(OBJECT_ID+7, Color.green, None, t2p(["A", "J", "B", "F", "C", "I", "M"])),
        ]
        OBJECT_ID += 8
        graph.add_nodes_from(nodes)
        for index in graph.node_indices():
            graph[index].index = index
        # insert edges between the nodes
        for node1, node2 in itertools.combinations(graph.nodes(), 2):
            if node1.color == node2.color or graph.has_edge(node1.index, node2.index):
                continue
            graph.add_edge(node1.index, node2.index, DualGraphEdge(OBJECT_ID, node1, node2))
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
        for r_colors in itertools.combinations(self.colors, 2):
            graphs[tuple(r_colors)] = {}
            graphs[tuple(r_colors[::-1])] = {}
            for m_color in set(self.colors) - set(r_colors):
                graph = self._construct_mc3_graph(r_colors, m_color)
                graphs[tuple(r_colors)][m_color] = graph
                graphs[tuple(r_colors[::-1])][m_color] = graph
        self._mc3_graphs = graphs
        return self._mc3_graphs[rcolors][monochromatic_color]

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
            graph.add_edge(node1.index, node2.index, Mc3GraphEdge(OBJECT_ID, node1, node2))
            OBJECT_ID += 1

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
        for r_colors in itertools.combinations(self.colors, 2):
            graphs[r_colors] = {}
            graphs[r_colors[::-1]] = {}
            for m_3_color in set(self.colors) - set(r_colors):
                m_4_color = (set(self.colors) - set(r_colors) - {m_3_color}).pop()
                graph = self._construct_mc4_graph(r_colors, m_3_color, m_4_color)
                graphs[r_colors][m_3_color] = graph
                graphs[r_colors[::-1]][m_3_color] = graph
        self._mc4_graphs = graphs
        return self._mc4_graphs[rcolors][monochromatic_3_color]

    def _construct_mc4_graph(self, restricted_colors: list[Color], monochromatic_3_color: Color, monochromatic_4_color: Color) -> rx.PyGraph:
        global OBJECT_ID
        graph = rx.PyGraph(multigraph=False)
        mc3_graph = self.mc3_graph(restricted_colors, monochromatic_3_color)
        dual_graph = self.dual_graph()
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
            graph.add_edge(node1.index, node2.index, Mc4GraphEdge(OBJECT_ID, node1, node2))
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
