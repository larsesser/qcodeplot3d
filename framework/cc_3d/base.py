import abc
import dataclasses
from dataclasses import dataclass
from typing import Optional

from framework.stabilizers import Stabilizer, Color


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


@dataclass
class DualGraphNode(GraphNode):
    """Representation of one node in the dual lattice.

    A node corresponds to a stabilizer or a boundary of the primary lattice.
    """
    qubits: list[int]
    is_stabilizer: bool
    stabilizer_length: Optional[int] = None
    stabilizer: Optional[Stabilizer] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.qubits = sorted(self.qubits)
        if not self.color.is_monochrome:
            raise ValueError
        if self.is_stabilizer and self.stabilizer_length is None:
            raise ValueError
        if self.is_stabilizer:
            self.stabilizer = Stabilizer(self.stabilizer_length, self.color, x_positions=self.qubits)

    @property
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
        if self.is_stabilizer and self.stabilizer_length is None:
            raise ValueError
        if self.is_stabilizer:
            self.stabilizer = Stabilizer(self.stabilizer_length, self.color, z_positions=self.qubits)


@dataclass
class DualGraphEdge(GraphEdge):
    node1: DualGraphNode
    node2: DualGraphNode
    stabilizer: Optional[Stabilizer] = dataclasses.field(init=False)

    def __post_init__(self):
        self._qubits = sorted(set(self.node1.qubits) & set(self.node2.qubits))
        if self.is_stabilizer:
            if self.node1.is_stabilizer:
                stab_length = self.node1.stabilizer.length
            else:
                stab_length = self.node2.stabilizer.length
            self.stabilizer = Stabilizer(length=stab_length, color=self.node1.color.combine(self.node2.color), z_positions=self._qubits)

    @property
    def qubits(self):
        """The qubits associated with this edge (== face in primal lattice)."""
        return self._qubits

    @property
    def is_stabilizer(self):
        """Is this edge (for a 3D color code) associated to a stabilizer?"""
        return not self.is_edge_between_boundaries


@dataclass
class XDualGraphEdge(DualGraphEdge):
    @property
    def is_stabilizer(self):
        return False
