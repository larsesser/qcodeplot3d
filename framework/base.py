import abc
import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Optional
import random

from framework.stabilizers import Color, Stabilizer


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
    # used for graph debugging
    title: Optional[str] = dataclasses.field(default=None, init=False)

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
    # weight: float = dataclasses.field(init=False)

    def __post_init__(self):
        _ = self.weight

    @cached_property
    def is_edge_between_boundaries(self) -> bool:
        return self.node1.is_boundary and self.node2.is_boundary

    @cached_property
    def has_boundary(self) -> bool:
        return self.node1.is_boundary or self.node2.is_boundary

    @cached_property
    def node_ids(self) -> tuple[int, int]:
        """Easier access of node ids (first lower, then higher id)."""
        if self.node1.id < self.node2.id:
            return self.node1.id, self.node2.id
        else:
            return self.node2.id, self.node1.id

    @cached_property
    def color(self) -> Optional[Color]:
        if self.node1.color.is_monochrome and self.node2.color.is_monochrome:
            return self.node1.color.combine(self.node2.color)
        elif self.node1.color.is_mixed and self.node2.color.is_mixed:
            return self.node1.color.intersect(self.node2.color)
        return None

    @cached_property
    def weight(self) -> float:
        return random.uniform(1 - 1e-6, 1 + 1e-6)

    def __contains__(self, item):
        """Boilerplate code for pymatching."""
        return False

    def get(self, attr, default):
        """Boilerplate code for pymatching."""
        if attr not in {"fault_ids", "weight", "error_probability"}:
            raise RuntimeError
        if attr == "weight":
            return self.weight
        return default


@dataclass
class DualGraphNode(GraphNode):
    """Representation of one node in the dual lattice.

    A node corresponds to a stabilizer or a boundary of the primary lattice.
    """
    id: int
    qubits: list[int]
    is_stabilizer: bool
    # all (physical) qubits of the color code where this node belongs to
    all_qubits: list[int] = dataclasses.field(repr=False)
    stabilizer: Optional[Stabilizer] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.qubits = sorted(self.qubits)
        if not self.color.is_monochrome:
            raise ValueError
        if self.is_stabilizer:
            # use id to ensure the ancilla of each stabilizer (node-like and edge-like!) is unique
            ancilla = len(self.all_qubits) + self.id + 1
            self.stabilizer = Stabilizer(len(self.all_qubits), self.color, x_positions=self.qubits, ancillas=[ancilla])

    @cached_property
    def is_boundary(self) -> bool:
        return not self.is_stabilizer


@dataclass
class XDualGraphNode(DualGraphNode):
    """Representation of one node in the x dual lattice.

    A node corresponds to an edge of the regular dual lattice (so a face of the primary lattice), which hosts either a
    stabilizer or is a boundary.
    """
    def __post_init__(self):
        self.qubits = sorted(self.qubits)
        if not self.color.is_mixed:
            raise ValueError
        if self.is_stabilizer:
            self.stabilizer = Stabilizer(len(self.all_qubits), self.color, z_positions=self.qubits)


@dataclass
class DualGraphEdge(GraphEdge):
    id: int
    node1: DualGraphNode
    node2: DualGraphNode
    stabilizer: Optional[Stabilizer] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        # super().__post_init__()
        if self.is_stabilizer:
            if self.node1.is_stabilizer:
                stab_length = self.node1.stabilizer.length
            else:
                stab_length = self.node2.stabilizer.length
            # use id to ensure the ancilla of each stabilizer (node-like and edge-like!) is unique
            ancilla = len(self.all_qubits) + self.id + 1
            self.stabilizer = Stabilizer(length=stab_length, color=self.node1.color.combine(self.node2.color), z_positions=self.qubits, ancillas=[ancilla])

    @cached_property
    def qubits(self) -> list[int]:
        """The qubits associated with this edge (== face in primal lattice)."""
        return sorted(set(self.node1.qubits) & set(self.node2.qubits))

    @cached_property
    def all_qubits(self) -> list[int]:
        """All (physical) qubits of the color code where this edge belongs to."""
        return self.node1.all_qubits

    @cached_property
    def is_stabilizer(self) -> bool:
        """Is this edge (for a 3D color code) associated to a stabilizer?"""
        return not self.is_edge_between_boundaries


@dataclass
class XDualGraphEdge(DualGraphEdge):
    @cached_property
    def is_stabilizer(self) -> bool:
        return False


@dataclass
class RestrictedGraphNode(GraphNode):
    """Representation of one node in a restricted lattice.

    A small wrapper around a DualGraphNode to get the indices right.
    """
    dg_node: DualGraphNode

    def __init__(self, dg_node: DualGraphNode) -> None:
        self.dg_node = dg_node

    @cached_property
    def color(self):
        return self.dg_node.color

    @cached_property
    def id(self):
        return self.dg_node.id

    @cached_property
    def is_boundary(self):
        return self.dg_node.is_boundary

    @cached_property
    def stabilizer(self):
        return self.dg_node.stabilizer

    @cached_property
    def qubits(self):
        return self.dg_node.qubits

    @cached_property
    def dg_nodes(self) -> list[DualGraphNode]:
        return [self.dg_node]

    @cached_property
    def dg_indices(self) -> set[int]:
        return {self.dg_node.index}


@dataclass
class RestrictedGraphEdge(GraphEdge):
    node1: RestrictedGraphNode
    node2: RestrictedGraphNode
    dg_edge: DualGraphEdge

    @cached_property
    def id(self):
        return self.dg_edge.id

    @cached_property
    def qubits(self):
        """The qubits associated with this edge (== face in primal lattice)."""
        return self.dg_edge.qubits
