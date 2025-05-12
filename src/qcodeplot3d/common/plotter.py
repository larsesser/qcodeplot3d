"""Plotting dual lattice & constructed primary lattice from graph definition."""
import abc
import dataclasses
import itertools
import pathlib
import re
from collections import defaultdict
from functools import cached_property
from tempfile import NamedTemporaryFile
from typing import ClassVar, Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista
import pyvista.plotting
import pyvista.plotting.themes
import rustworkx as rx
import vtk
from rustworkx.visualization import graphviz_draw

from qcodeplot3d.common import compute_simplexes
from qcodeplot3d.common.graph import DualGraphNode, GraphEdge, GraphNode
from qcodeplot3d.common.stabilizers import Color

# see https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydata.n_faces#pyvista.PolyData.n_faces
pyvista.PolyData.use_strict_n_faces(True)


def convert_faces(faces: list[list[int]]) -> list[int]:
    """Pad a list of faces so that pyvista can process it."""
    return np.asarray(list(itertools.chain.from_iterable([(len(face), *face) for face in faces])), np.integer)


def reconvert_faces(faces: list[int]) -> list[list[int]]:
    ret = []
    iterator = iter(faces)
    while True:
        try:
            next_face_length = next(iterator)
        except StopIteration:
            break
        ret.append([next(iterator) for _ in range(next_face_length)])
    return ret


def triangles_to_face(triangles: list[list[int]]) -> list[int]:
    """Converts a triangulation of a 2D plane to a single plane.

    :param triangles: list of all simplexes (triangles) of the plane.
    """
    if any(len(triangle) != 3 for triangle in triangles):
        raise ValueError

    # Each edge "inside" the plane is present exactly twice, while each edge
    # at the boundary of the plane is present exactly once.
    boundary_edges = set()
    for triangle in triangles:
        for a, b in itertools.combinations(triangle, 2):
            if a < b:
                edge = (a, b)
            else:
                edge = (b, a)
            if edge in boundary_edges:
                boundary_edges.remove(edge)
            else:
                boundary_edges.add(edge)

    # Each point of the plane is present in exactly two different boundary edges.
    start, _ = boundary_edges.pop()
    face = [start]
    while boundary_edges:
        for edge in boundary_edges:
            a, b = edge
            if face[-1] == a:
                face.append(b)
                break
            if face[-1] == b:
                face.append(a)
                break
        boundary_edges.remove(edge)
    if len(set(face)) != len(face):
        raise RuntimeError(face)
    return face


def project_to_2d_plane(points: list[list[float]]) -> list[list[float]]:
    """Take a bunch of 3D points laying in (approximately) one 2D plane and project them to 2D points wrt this plane."""
    if any(len(point) != 3 for point in points):
        raise ValueError("All points must have 3D coordinates.")
    if len(points) < 3:
        raise ValueError("Need at least 3 points to determine the plane.")
    points = np.asarray(points)
    p_transposed = points.transpose()

    # paragraph taken from https://math.stackexchange.com/a/99317
    # "center of mass" of the plane
    centeroid = np.mean(p_transposed, axis=1, keepdims=True)
    # calculate the singular value decomposition of the centered points
    svd = np.linalg.svd(p_transposed - centeroid)
    # the left singular vector is the searched normal vector
    normal = svd[0][:, -1]

    # calculate the new base vectors of the 2D plane
    ex = None
    for a, b in itertools.combinations(points, 2):
        # choose points which do not share any coordinates
        if any(coordinate == 0.0 for coordinate in a-b):
            continue
        ex = (a-b) / np.abs(a-b)
        break
    if ex is None:
        raise RuntimeError(points)
    ey = np.cross(normal, ex) / np.abs(np.cross(normal, ex))

    # project the points on the plane
    projected = [p - (np.dot(p, normal) / np.dot(normal, normal)) * normal for p in points]
    ret = [[np.dot(p, ex), np.dot(p, ey)] for p in projected]

    return ret


def project_to_3d_plane(points: list[list[float]]) -> list[list[float]]:
    """Take a bunch of 3D points laying in (approximately) one 2D plane and project them to 2D points wrt this plane."""
    if any(len(point) != 3 for point in points):
        raise ValueError("All points must have 3D coordinates.")
    if len(points) < 3:
        raise ValueError("Need at least 3 points to determine the plane.")
    points = np.asarray(points)
    p_transposed = points.transpose()

    # paragraph taken from https://math.stackexchange.com/a/99317
    # "center of mass" of the plane
    centeroid = np.mean(p_transposed, axis=1, keepdims=True)
    # calculate the singular value decomposition of the centered points
    svd = np.linalg.svd(p_transposed - centeroid)
    # the left singular vector is the searched normal vector
    normal = svd[0][:, -1]
    normal = normal / np.linalg.norm(normal)

    centeroid = centeroid.transpose()[0]
    projected = [p - (p - centeroid).dot(normal) for p in points]

    return projected

def project_to_given_plane(plane: list[np.ndarray], points: list[list[float]]) -> list[np.ndarray]:
    """Take a bunch of 3D points laying and a 3D plane and project them to the 3D plane."""
    if any(len(point) != 3 for point in points):
        raise ValueError("All points must have 3D coordinates.")
    if len(plane) < 3:
        raise ValueError("Need at least 3 points to determine the plane.")
    points = np.asarray(points)

    n = np.cross(plane[0] - plane[1], plane[0] - plane[2])
    n_normalized = n / np.linalg.norm(n)
    ret = []
    for point in points:
        t = (np.dot(n_normalized, plane[0]) - np.dot(n_normalized, point)) / np.dot(n_normalized, n_normalized)
        ret.append(point + t * n_normalized)

    return ret


def cross_point_3_planes(plane1: list[np.ndarray], plane2: list[np.ndarray], plane3: list[np.ndarray]) -> list[np.ndarray]:
    n_planes: list[npt.NDArray[np.float64]] = []
    b_planes: list[float] = []

    for p in [plane1, plane2, plane3]:
        points = np.asarray(p)
        p_transposed = points.transpose()
        # paragraph taken from https://math.stackexchange.com/a/99317
        # "center of mass" of the plane
        centeroid = np.mean(p_transposed, axis=1, keepdims=True)
        a = centeroid.transpose()[0]
        # calculate the singular value decomposition of the centered points
        svd = np.linalg.svd(p_transposed - centeroid)
        # the left singular vector is the searched normal vector
        normal = svd[0][:, -1]
        b = a.dot(normal)
        n_planes.append(normal)
        b_planes.append(b)

    # treat plane equations as system of linear equations, solve to obtain cross point
    return np.linalg.solve(n_planes, b_planes)


def project_to_line(line: list[np.ndarray], point: np.ndarray) -> np.ndarray:
    """from https://stackoverflow.com/a/61343727"""
    p1, p2 = line[0], line[1]
    # distance between p1 and p2
    l2 = np.sum((p1 - p2) ** 2)
    if l2 == 0:
        print('p1 and p2 are the same points')

    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(point-p1) . (p2-p1)] / |p2-p1|^2

    # if you need the point to project on line extention connecting p1 and p2
    t = np.sum((point - p1) * (p2 - p1)) / l2

    # if you need to ignore if p3 does not project onto line segment
    # if t > 1 or t < 0:
    #     print('p3 does not project onto p1-p2 line segment')

    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    # t = max(0, min(1, np.sum((point - p1) * (p2 - p1)) / l2))

    return p1 + t * (p2 - p1)


def cross_point_2_lines(line1: list[np.ndarray], line2: list[np.ndarray]) -> np.ndarray:
    """Line1 and Line2 are lists of two or more points on the respecitve line."""
    a = []
    b = []
    for i in [0, 1, 2]:
        a.append([line1[0][i] - line1[1][i], line2[0][i] - line2[1][i]])
        b.append(- line1[0][i] + line2[0][i])
    s = np.linalg.lstsq(a, b)
    return line1[0] + s[0][0] * (line1[0] - line1[1])


def distance_to_plane(plane: list[np.ndarray], points: list[list[np.ndarray]]) -> list[float]:
    n = np.cross(plane[0] - plane[1], plane[0] - plane[2])
    n_normalized = n / np.linalg.norm(n)
    return [np.abs(np.dot(n_normalized, point - plane[0])) for point in points]


def distance_between_points(point1: list[np.ndarray], point2: list[np.ndarray]) -> float:
    return np.sqrt(sum((a-b)**2 for a, b in zip(point1, point2)))


@dataclasses.dataclass
class Plotter(abc.ABC):
    dual_graph: rx.PyGraph
    distance: int
    storage_dir: pathlib.Path = dataclasses.field(default=pathlib.Path(__file__).parent.parent.absolute())
    highes_id: int = dataclasses.field(default=0, init=False)
    _dualgraph_to_dualmesh: dict[int, int] = dataclasses.field(default=None, init=False)
    _dualmesh_to_dualgraph: dict[int, int] = dataclasses.field(default=None, init=False)
    dimension: ClassVar[int] = 0

    @cached_property
    def boundary_nodes(self) -> list[DualGraphNode]:
        return [node for node in self.dual_graph.nodes() if node.is_boundary]

    @cached_property
    def dual_mesh(self) -> pyvista.PolyData:
        return self._construct_dual_mesh()

    @cached_property
    def qubit_to_boundaries(self) -> dict[int, list[DualGraphNode]]:
        ret: dict[int, list[DualGraphNode]] = dict()

        # group dual lattice cells (of the tetrahedron) by qubit
        qubit_to_facepositions: dict[int, list[int]] = defaultdict(list)
        for position, qubit in enumerate(self.dual_mesh.cell_data['qubits']):
            qubit_to_facepositions[qubit].append(position)

        dual_mesh_faces = reconvert_faces(self.dual_mesh.faces)
        for pointposition, (qubit, facepositions) in enumerate(qubit_to_facepositions.items()):
            dg_nodes = [self.get_dual_node(point_index) for point_index in
                        sorted(set().union(*[dual_mesh_faces[face_index] for face_index in facepositions]))]
            ret[qubit] = [node for node in dg_nodes if node.is_boundary]
        return ret

    @staticmethod
    def get_plotting_theme() -> pyvista.plotting.themes.DocumentTheme:
        theme = pyvista.plotting.themes.DocumentTheme()
        theme.cmap = Color.highlighted_color_map()
        theme.show_vertices = False
        theme.show_edges = True
        theme.lighting = 'light kit'
        theme.render_points_as_spheres = True
        theme.render_lines_as_tubes = True
        theme.hidden_line_removal = False
        return theme

    @property
    def next_id(self) -> int:
        """Generate the next unique id to label mesh objects."""
        self.highes_id += 1
        return self.highes_id

    def get_dual_node(self, mesh_index: int) -> DualGraphNode:
        return self.dual_graph[self._dualmesh_to_dualgraph[mesh_index]]

    def get_dual_mesh_index(self, graph_index: int) -> int:
        return self._dualgraph_to_dualmesh[graph_index]

    def layout_rustworkx_graph(self, graph: rx.PyGraph) -> dict[int, npt.NDArray[np.float64]]:
        """Calculate 3D coordinates of nodes by layouting the rustworkx graph.

        Take special care to place boundary nodes at a meaningful position.

        :returns: Mapping of rx node indices to [x, y, z] coordinates.
        """
        # remove boundary nodes for bulk positioning
        graph_without_boundaries = graph.copy()
        boundary_node_indices = [node.index for node in graph_without_boundaries.nodes() if node.is_boundary]
        graph_without_boundaries.remove_nodes_from(boundary_node_indices)
        # if there is only one bulk node, we can't perform the normal layouting algorithm
        no_boundary_special_handling = len(graph_without_boundaries.nodes()) == 1
        if no_boundary_special_handling:
            graph_without_boundaries = graph

        with NamedTemporaryFile("w+t", suffix=".wrl") as f:
            graphviz_draw(graph_without_boundaries, lambda node: {"shape": "point"}, filename=f.name, method="neato",
                          image_type="vrml", graph_attr={"dimen": f"{self.dimension}"})
            data = f.readlines()

        # position of non-boundary nodes
        ret: dict[int, npt.NDArray[np.float64]] = {}
        node_pattern = re.compile(r"^# node (?P<node_index>\d+)$")
        pos_pattern = re.compile(r"^\s*translation (?P<x>-?\d+.\d+) (?P<y>-?\d+.\d+) (?P<z>-?\d+.\d+)$")
        for line_nr, line in enumerate(data):
            match = re.match(node_pattern, line)
            if match is None:
                continue
            node_index = int(match.group("node_index"))
            pos_match = re.match(pos_pattern, data[line_nr + 2])
            if pos_match is None:
                print(data[line_nr + 2])
                raise RuntimeError(node_index, line_nr)
            x = float(pos_match.group("x"))
            y = float(pos_match.group("y"))
            z = float(pos_match.group("z"))
            ret[node_index] = np.asarray([x, y, z])

        if no_boundary_special_handling:
            return ret

        # center position of the bulk
        center = np.asarray([0.0, 0.0, 0.0])
        for position in ret.values():
            center += position
        center /= len(ret)

        # position of boundary nodes
        for boundary_index in boundary_node_indices:
            adjacent_nodes = [index for index in graph.neighbors(boundary_index) if not graph[index].is_boundary]
            face_center = np.asarray([0.0, 0.0, 0.0])
            for index, position in ret.items():
                if index not in adjacent_nodes:
                    continue
                face_center += position
            face_center /= len(adjacent_nodes)

            # extrapolate the position of the boundary node from the line through center and face_center
            pos = face_center + 1*(face_center - center)
            ret[boundary_index] = pos
        return ret

    def _construct_dual_mesh(self, highlighted_nodes: list[GraphNode] = None) -> pyvista.PolyData:
        highlighted_nodes = highlighted_nodes or []
        # calculate positions of points
        node2coordinates = self.layout_rustworkx_graph(self.dual_graph)
        points = np.asarray([node2coordinates[index] for index in self.dual_graph.node_indices()])

        # generate pyvista edges from rustworkx edges
        rustworkx2pyvista = {rustworkx_index: pyvista_index for pyvista_index, rustworkx_index in enumerate(self.dual_graph.node_indices())}
        self._dualgraph_to_dualmesh = rustworkx2pyvista
        self._dualmesh_to_dualgraph = {value: key for key, value in rustworkx2pyvista.items()}
        # TODO ensure all faces of dual graph are triangles?
        simplexes = compute_simplexes(self.dual_graph, dimension=self.dimension, exclude_boundary_simplexes=True)
        if self.dimension == 2:
            faces = [[rustworkx2pyvista[index] for index in simplex] for simplex in simplexes]
        elif self.dimension == 3:
            # each simplex (tetrahedron) has four faces (triangles)
            faces = [[rustworkx2pyvista[index] for index in combination]
                     for simplex in simplexes for combination in itertools.combinations(simplex, 3)]
        else:
            raise NotImplementedError

        ret = pyvista.PolyData(points, faces=convert_faces(faces))

        # add point labels
        point_labels = []
        for node in self.dual_graph.nodes():
            label = f"{node.index}"
            if node.title:
                label = f"{node.title}"
            if node.is_boundary:
                label += " B"
            point_labels.append(label)
        ret["point_labels"] = point_labels

        # add tetrahedron ids
        # TODO qubit als tetrahedron id nutzen?
        tetrahedron_ids = itertools.chain.from_iterable([[self.next_id]* (4 if self.dimension == 3 else 1) for _ in simplexes])
        ret.cell_data["face_ids"] = list(tetrahedron_ids)

        # add the qubit to each face of its tetrahedron
        labels = []
        for simplex in simplexes:
            qubits = set(self.dual_graph.nodes()[simplex[0]].qubits)
            for index in simplex[1:]:
                qubits &= set(self.dual_graph.nodes()[index].qubits)
            if len(qubits) != 1:
                raise RuntimeError
            labels.extend([qubits.pop()] * (4 if self.dimension == 3 else 1))
        ret.cell_data["qubits"] = labels

        # add color
        colors = [node.color.highlight if node in highlighted_nodes else node.color for node in self.dual_graph.nodes()]
        ret["colors"] = colors

        return ret

    def construct_dual_mesh(
        self,
        graph: rx.PyGraph,
        *,
        coordinates: dict[int, npt.NDArray[np.float64]] = None,
        use_edges_colors: bool = False,
        edge_color: Color = None,
        highlighted_nodes: list[GraphNode] = None,
        highlighted_edges: list[GraphEdge] = None,
        include_edges_between_boundaries: bool = True,
        exclude_boundaries: bool = False,
        mandatory_qubits: set[int] = None,
    ) -> pyvista.PolyData:
        coordinates = coordinates or self.layout_dual_nodes(self.construct_primary_mesh())
        return self.construct_debug_mesh(
            graph,
            coordinates=coordinates,
            use_edges_colors=use_edges_colors,
            edge_color=edge_color,
            highlighted_nodes=highlighted_nodes,
            highlighted_edges=highlighted_edges,
            include_edges_between_boundaries=include_edges_between_boundaries,
            exclude_boundaries=exclude_boundaries,
            mandatory_qubits=mandatory_qubits,
        )

    def construct_debug_mesh(
        self,
        graph: rx.PyGraph,
        *,
        coordinates: dict[int, npt.NDArray[np.float64]] = None,
        use_edges_colors: bool = False,
        edge_color: Color = None,
        highlighted_nodes: list[GraphNode] = None,
        highlighted_edges: list[GraphEdge] = None,
        include_edges_between_boundaries: bool = True,
        exclude_boundaries: bool = False,
        mandatory_qubits: set[int] = None,
    ) -> pyvista.PolyData:
        """Create a 3D mesh of the given rustworkx Graph.

        Nodes must be GraphNode and edges GraphEdge objects.

        :param coordinates: Mapping of node indices of graph to 3D coordinates. Use them instead of calculating them.
        :param use_edges_colors: If true, use the color of the GraphEdge object instead of default colors.
        :param highlighted_nodes: Change color of given nodes to highlighted color.
        """
        if mandatory_qubits:
            graph = graph.copy()
            for node in graph.nodes():
                if not set(node.qubits) & mandatory_qubits:
                    graph.remove_node(node.index)
        if exclude_boundaries:
            graph = graph.copy()
            for node in graph.nodes():
                if node.is_boundary:
                    graph.remove_node(node.index)
        highlighted_nodes = highlighted_nodes or []
        highlighted_edges = highlighted_edges or []

        # calculate positions of points (or use given coordinates)
        node2coordinates = coordinates or self.layout_rustworkx_graph(graph)
        points = np.asarray([node2coordinates[index] for index in graph.node_indices()])

        # generate pyvista edges from rustworkx edges
        rustworkx2pyvista = {rustworkx_index: pyvista_index for pyvista_index, rustworkx_index in enumerate(graph.node_indices())}
        lines = [[rustworkx2pyvista[edge.node1.index], rustworkx2pyvista[edge.node2.index]] for edge in graph.edges()
                 if include_edges_between_boundaries or not edge.is_edge_between_boundaries]
        ret = pyvista.PolyData(points, lines=convert_faces(lines))

        # remember which nodes are boundary nodes
        boundaries = [node.is_boundary for node in graph.nodes()]
        ret["is_boundary"] = boundaries

        # remember the edge index and weight
        ret.cell_data["edge_index"] = [
            edge.index
            for edge in graph.edges()
            if include_edges_between_boundaries or not edge.is_edge_between_boundaries
        ]
        ret.cell_data["edge_weight"] = [
            np.round(getattr(edge, "weight", -1), decimals=3)
            for edge in graph.edges()
            if include_edges_between_boundaries or not edge.is_edge_between_boundaries
        ]

        # add point labels
        point_labels = []
        for node in graph.nodes():
            label = ""
            if node.title:
                label = f"{node.title}"
            elif node.id:
                label = f"{node.id}"
            if node.is_boundary:
                label += " B"
            point_labels.append(label)
        ret["point_labels"] = point_labels

        # add colors to lines
        edge_colors = []
        for edge in graph.edges():
            if edge_color:
                color = edge_color
            elif use_edges_colors:
                # use grey as fallback
                color = Color.by
                if edge.color is not None:
                    if self.dimension == 2:
                        color = ({Color.red, Color.green, Color.blue} - {edge.node1.color, edge.node2.color}).pop()
                    elif self.dimension == 3:
                        color = edge.color
            elif edge.is_edge_between_boundaries:
                color = Color.red
            # elif edge.node1.is_boundary or edge.node2.is_boundary:
            #     color = Color.green
            else:
                # grey
                color = Color.by
            if edge in highlighted_edges:
                color = color.highlight
            if edge.is_edge_between_boundaries and not include_edges_between_boundaries:
                continue
            edge_colors.append(color)
        ret.cell_data["edge_colors"] = edge_colors

        # add colors to points
        colors = [node.color.highlight if node in highlighted_nodes else node.color for node in graph.nodes()]
        ret.point_data["colors"] = colors

        return ret

    @abc.abstractmethod
    def construct_primary_mesh(self, highlighted_volumes: list[DualGraphNode] = None,
                               qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                               face_color: Union[Color, list[Color]] = None, node_color: Union[Color, list[Color]] = None,
                               lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                               mandatory_face_qubits: set[int] = None, string_operator_qubits: set[int] = None,
                               color_edges: bool = False, mandatory_cell_qubits: set[int] = None,
                               face_syndrome_qubits: set[int] = None) -> pyvista.PolyData:
        """Construct primary mesh from dual_mesh.

        :param qubit_coordinates: Use them instead of calculating the coordinates from the dual_mesh.
        :param face_color: If present, show only faces with this color.
        :param node_color: If present, show only nodes with this color.
        :param lowest_title: If present, only nodes with col, row, layer higher or equal than this tuple are shown.
        :param highest_title: If present, only nodes with col, row, layer lower or equal than this tuple are shown.
        :param mandatory_face_qubits: If present, only faces with support on any of this qubits are shown.
        :param string_operator_qubits: If present, only edges with this qubits will be displayed. Mainly useful if
            only_nodes_with_color is given.
        :param color_edges: If true, color edges with the color of the cells they connect.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def _layout_dual_nodes_factor(distance: int) -> Optional[float]:
        ...

    def layout_dual_nodes(self, primary_mesh: pyvista.PolyData) -> dict[int, npt.NDArray[np.float64]]:
        # compute the center of each volume
        ret: dict[int, npt.NDArray[np.float64]] = self.preprocess_dual_node_layout(primary_mesh)

        qubit_points: dict[int, npt.NDArray[np.float64]] = {
            primary_mesh.point_data['qubits'][pos]: point for pos, point in enumerate(primary_mesh.points)}
        corner_qubits = {qubit for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == self.dimension}

        # calculate the center of the code
        center = np.asarray([0.0, 0.0, 0.0])
        for qubit in corner_qubits:
            center += qubit_points[qubit]
        center = center / len(corner_qubits)

        # compute the position of each boundary node
        for node in self.boundary_nodes:
            qubits = corner_qubits & set(node.qubits)
            face_center = np.asarray([0.0, 0.0, 0.0])
            for qubit in qubits:
                face_center += qubit_points[qubit]
            face_center = face_center / len(qubits)

            # extrapolate the position of the boundary node from the line through center and face_center
            factor = self._layout_dual_nodes_factor(self.distance) or 1
            ret[node.index] = face_center + factor*(face_center - center)

        return ret

    @abc.abstractmethod
    def preprocess_dual_node_layout(self, primary_mesh: pyvista.PolyData):
        ...

    @staticmethod
    def get_qubit_coordinates(primary_mesh: pyvista.PolyData) -> dict[int, npt.NDArray[np.float64]]:
        return {qubit: coordinate for qubit, coordinate in zip(primary_mesh.point_data['qubits'], primary_mesh.points)}

    def plot_debug_mesh(
        self,
        mesh: pyvista.PolyData,
        *,
        show_labels: bool = False,
        point_size: int = None,
        line_width: int = None,
        edge_color: str = None,
        camera_position: list[tuple[float, float, float]] = None,
        print_camera_position: bool = False,
        filename: pathlib.Path = None,
        window_title: str = None,
    ) -> None:
        # use default values
        if point_size is None:
            point_size = 15 if filename is None else 120
        if line_width is None:
            line_width = 3 if filename is None else 10

        plt = pyvista.plotting.Plotter(theme=self.get_plotting_theme(), off_screen=filename is not None)
        if show_labels:
            plt.add_point_labels(mesh, "point_labels", point_size=point_size, font_size=20)
        if edge_color:
            plt.add_mesh(mesh, show_scalar_bar=False, color=edge_color, line_width=line_width, smooth_shading=True,
                         show_vertices=True, point_size=point_size, style="wireframe")
        else:
            plt.add_mesh(mesh, scalars="edge_colors", show_scalar_bar=False, cmap=Color.color_map(),
                         clim=Color.color_limits(), line_width=line_width, smooth_shading=True,
                         show_vertices=True, point_size=point_size, style="wireframe")
        plt.add_points(mesh.points, scalars=mesh["colors"], point_size=point_size, show_scalar_bar=False,
                       clim=Color.color_limits())

        light = pyvista.Light(light_type='headlight')
        light.intensity = 0.8
        plt.add_light(light)

        if camera_position:
            plt.camera_position = camera_position
        else:
            plt.reset_camera()
            plt.set_focus(mesh.center)
        plt.camera_set = True
        if filename is None:
            if print_camera_position:
                plt.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, lambda *args: print(str(plt.camera_position)))
            plt.show(title=window_title)
        else:
            plt.screenshot(filename=str(filename), scale=5)

    def plot_primary_mesh(
        self,
        *,
        show_qubit_labels: bool = False,
        qubit_point_size: int = None,
        highlighted_qubit_point_size: int = None,
        highlighted_volumes: list[DualGraphNode] = None,
        highlighted_qubits: list[int] = None,
        qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
        only_faces_with_color: Union[Color, list[Color]] = None,
        only_nodes_with_color: Union[Color, list[Color]] = None,
        lowest_title: tuple[int, int, int] = None,
        highest_title: tuple[int, int, int] = None,
        mandatory_face_qubits: set[int] = None,
        string_operator_qubits: set[int] = None,
        color_edges: bool = True,
        show_normal_qubits: bool = True,
        line_width: int = None,
        transparent_faces: bool = False,
        mandatory_cell_qubits: set[int] = None,
        face_syndrome_qubits: set[int] = None,
        camera_position: list[tuple[float, float, float]] = None,
        print_camera_position: bool = False,
        filename: pathlib.Path = None,
        window_title: str = None,
    ) -> None:
        # set default values
        if qubit_point_size is None:
            qubit_point_size = 15 if filename is None else 70
        if highlighted_qubit_point_size is None:
            highlighted_qubit_point_size = 17 if filename is None else 100
        if line_width is None:
            line_width = 3 if filename is None else 10

        plt = self._plot_primary_mesh_internal(
            show_qubit_labels=show_qubit_labels,
            qubit_point_size=qubit_point_size,
            highlighted_qubit_point_size=highlighted_qubit_point_size,
            highlighted_volumes=highlighted_volumes,
            highlighted_qubits=highlighted_qubits,
            qubit_coordinates=qubit_coordinates,
            only_faces_with_color=only_faces_with_color,
            only_nodes_with_color=only_nodes_with_color,
            lowest_title=lowest_title,
            highest_title=highest_title,
            mandatory_face_qubits=mandatory_face_qubits,
            string_operator_qubits=string_operator_qubits,
            color_edges=color_edges,
            show_normal_qubits=show_normal_qubits,
            line_width=line_width,
            transparent_faces=transparent_faces,
            mandatory_cell_qubits=mandatory_cell_qubits,
            face_syndrome_qubits=face_syndrome_qubits,
            off_screen=filename is not None,
        )
        plt.camera_position = camera_position
        if filename is None:
            if print_camera_position:
                plt.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, lambda *args: print(str(plt.camera_position)))
            plt.show(title=window_title)
        else:
            plt.screenshot(filename=str(filename), scale=5)

    def _plot_primary_mesh_internal(
        self,
        *,
        show_qubit_labels: bool = False,
        qubit_point_size: int = None,
        highlighted_qubit_point_size: int = None,
        highlighted_volumes: list[DualGraphNode] = None,
        highlighted_qubits: list[int] = None,
        qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
        only_faces_with_color: Union[Color, list[Color]] = None,
        only_nodes_with_color: Union[Color, list[Color]] = None,
        lowest_title: tuple[int, int, int] = None,
        highest_title: tuple[int, int, int] = None,
        mandatory_face_qubits: set[int] = None,
        string_operator_qubits: set[int] = None,
        color_edges: bool = True,
        show_normal_qubits: bool = True,
        line_width: int = 1,
        transparent_faces: bool = False,
        mandatory_cell_qubits: set[int] = None,
        face_syndrome_qubits: set[int] = None,
        off_screen: bool = None,
    ) -> pyvista.plotting.Plotter:
        """Return the plotter preloaded with the primary mesh."""
        # set default values
        highlighted_qubits = highlighted_qubits or []

        # TODO add caching for specific call combinations? or for plain combination?
        mesh = self.construct_primary_mesh(highlighted_volumes, qubit_coordinates, only_faces_with_color,
                                           only_nodes_with_color, lowest_title, highest_title, mandatory_face_qubits,
                                           string_operator_qubits, color_edges, mandatory_cell_qubits, face_syndrome_qubits)

        theme = self.get_plotting_theme()
        if self.dimension == 3 and not transparent_faces:
            theme.cmap = Color.highlighted_color_map_3d()

        plt = pyvista.plotting.Plotter(theme=theme, off_screen=off_screen)
        # extract lines from mesh, plot them separately
        if only_nodes_with_color is not None or color_edges:
            line_poses = reconvert_faces(mesh.lines)
            edge_mesh = pyvista.PolyData(mesh.points, lines=mesh.lines)
            if only_nodes_with_color is not None:
                line_color = Color.highlighted_color_map_3d().colors[Color(only_nodes_with_color).highlight]
                plt.add_mesh(edge_mesh, show_scalar_bar=False, line_width=25, smooth_shading=True,
                             color=line_color, point_size=qubit_point_size, show_vertices=False, style="wireframe")
            elif color_edges:
                if line_width is None:
                    raise ValueError(line_width)
                edge_mesh.cell_data["colors"] = list(mesh.cell_data["colors"])[:len(line_poses)]
                plt.add_mesh(edge_mesh, show_scalar_bar=False, line_width=line_width, smooth_shading=True,
                             clim=Color.color_limits(), scalars="colors", point_size=qubit_point_size,
                             show_vertices=False, style="wireframe")
            # remove lines from mesh
            mesh.lines = None
            mesh.cell_data["colors"] = list(mesh.cell_data["colors"])[len(line_poses):]

        # only show qubits which are present in at least one face
        used_qubit_pos = set(itertools.chain.from_iterable(reconvert_faces(mesh.faces)))
        if string_operator_qubits:
            used_qubit_pos.update([pos for pos, qubit in enumerate(mesh.point_data['qubits']) if qubit in string_operator_qubits])
        if show_normal_qubits:
            normal_qubits = set(mesh.point_data['qubits']) - set(highlighted_qubits)
        else:
            normal_qubits = set()
        for qubits, color, point_size in [
                (normal_qubits, "indigo", qubit_point_size),
                (highlighted_qubits, "violet", highlighted_qubit_point_size),
            ]:
            positions = [pos for pos, qubit in enumerate(mesh.point_data['qubits'])
                         if qubit in qubits and (pos in used_qubit_pos or face_syndrome_qubits)]
            coordinates = np.asarray([coordinate for pos, coordinate in enumerate(mesh.points) if pos in positions])
            if len(coordinates) == 0:
                continue
            plt.add_points(coordinates, point_size=point_size, color=color)
            if show_qubit_labels:
                qubit_labels = [f"{qubit}" for pos, qubit in enumerate(mesh.point_data['qubits']) if pos in positions]
                plt.add_point_labels(coordinates, qubit_labels, show_points=False, font_size=20)

        if not color_edges:
            plt.add_mesh(mesh, show_scalar_bar=False, color="black", smooth_shading=True, line_width=line_width, style='wireframe')
        plt.add_mesh(mesh, scalars="colors", show_scalar_bar=False, clim=Color.color_limits(), smooth_shading=True,
                     show_edges=False, opacity=0.2 if transparent_faces else None,
                     ambient=1 if transparent_faces else None, diffuse=0 if transparent_faces else None)
        plt.reset_camera()
        plt.set_focus(mesh.center)
        plt.camera_set = True

        # useful code sniped to print all qubits of all faces which lay in the same plane, given by a face of qubits
        # plane_qubits = [78, 1388, 466]
        # req_face_color = Color.rb
        # plane_pos = [pos for pos, qubit in enumerate(mesh.point_data['qubits']) if qubit in plane_qubits]
        # plane = [point for pos, point in enumerate(mesh.points) if pos in plane_pos]
        # stored_qubits = set()
        # for face, face_color in zip(reconvert_faces(mesh.faces), mesh.cell_data["colors"]):
        #     points = [point for pos, point in enumerate(mesh.points) if pos in face]
        #     if any(d < 10 for d in distance_to_plane(plane, points)) and face_color == req_face_color:
        #         stored_qubits.update([qubit for pos, qubit in enumerate(mesh.point_data['qubits']) if pos in face])
        # print(sorted(stored_qubits))
        # exit()

        light = pyvista.Light(light_type='headlight')
        light.intensity = 0.8
        plt.add_light(light)
        if self.dimension == 3 and not transparent_faces:
            plt.remove_all_lights()

        return plt

    def plot_debug_primary_mesh(
        self,
        mesh: pyvista.PolyData,
        *,
        show_qubit_labels: bool = False,
        highlighted_volumes: list[DualGraphNode] = None,
        highlighted_qubits: list[int] = None,
        qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
        only_faces_with_color: Union[Color, list[Color]] = None,
        only_nodes_with_color: Union[Color, list[Color]] = None,
        lowest_title: tuple[int, int, int] = None,
        highest_title: tuple[int, int, int] = None,
        mandatory_face_qubits: set[int] = None,
        string_operator_qubits: set[int] = None,
        color_edges: bool = True,
        show_normal_qubits: bool = True,
        transparent_faces: bool = False,
        highlighted_edges: list[GraphEdge] = None,
        qubit_point_size: int = None,
        highlighted_qubit_point_size: int = None,
        mesh_line_width: int = None,
        node_point_size: int = None,
        show_normal_edges: bool = True,
        primary_line_width: int = None,
        highlighted_line_width: int = None,
        mesh_line_color: Optional[str] = None,
        mandatory_cell_qubits: set[int] = None,
        face_syndrome_qubits: set[int] = None,
        show_edge_weights: bool = False,
        camera_position: list[tuple[float, float, float]] = None,
        print_camera_position: bool = False,
        filename: pathlib.Path = None,
        window_title: str = None,
    ) -> None:
        """Return the plotter preloaded with the debug and primary mesh."""
        # use default values
        if qubit_point_size is None:
            qubit_point_size = 15 if filename is None else 70
        if highlighted_qubit_point_size is None:
            highlighted_qubit_point_size = 17 if filename is None else 100
        if node_point_size is None:
            node_point_size = 20 if filename is None else 120
        if mesh_line_width is None:
            mesh_line_width = 3 if filename is None else 10
        if primary_line_width is None:
            primary_line_width = 3 if filename is None else 10
        if highlighted_line_width is None:
            highlighted_line_width = 5 if filename is None else 25

        plt = self._plot_primary_mesh_internal(
            show_qubit_labels=show_qubit_labels,
            qubit_point_size=qubit_point_size,
            highlighted_qubit_point_size=highlighted_qubit_point_size,
            highlighted_volumes=highlighted_volumes,
            highlighted_qubits=highlighted_qubits,
            qubit_coordinates=qubit_coordinates,
            only_faces_with_color=only_faces_with_color,
            only_nodes_with_color=only_nodes_with_color,
            lowest_title=lowest_title,
            highest_title=highest_title,
            mandatory_face_qubits=mandatory_face_qubits,
            string_operator_qubits=string_operator_qubits,
            color_edges=color_edges,
            show_normal_qubits=show_normal_qubits,
            line_width=primary_line_width,
            transparent_faces=transparent_faces,
            mandatory_cell_qubits=mandatory_cell_qubits,
            face_syndrome_qubits=face_syndrome_qubits,
            off_screen=filename is not None,
        )

        if highlighted_edges:
            highlighted_edge_indices = [edge.index for edge in highlighted_edges]
            all_edges = reconvert_faces(mesh.lines)
            normal_edge_pos = [pos for pos, index in enumerate(mesh.cell_data['edge_index']) if index not in highlighted_edge_indices]
            if normal_edge_pos and show_normal_edges:
                normal_edge = pyvista.PolyData(
                    mesh.points, lines=convert_faces([edge for pos, edge in enumerate(all_edges) if pos in normal_edge_pos]))
                plt.add_mesh(normal_edge, show_scalar_bar=False, line_width=mesh_line_width, smooth_shading=True, color="silver",
                             point_size=node_point_size, show_vertices=True, style="wireframe")
            highlighted_edge_pos = [pos for pos, index in enumerate(mesh.cell_data['edge_index']) if index in highlighted_edge_indices]
            highlighted_edge = pyvista.PolyData(
                mesh.points, lines=convert_faces([edge for pos, edge in enumerate(all_edges) if pos in highlighted_edge_pos]))
            plt.add_mesh(highlighted_edge, show_scalar_bar=False, line_width=highlighted_line_width, smooth_shading=True, color="orange",
                         point_size=node_point_size, show_vertices=True, style="wireframe")
        elif show_normal_edges:
            if mesh_line_color:
                plt.add_mesh(mesh, show_scalar_bar=False, point_size=node_point_size, line_width=mesh_line_width, smooth_shading=True,
                            color="silver", show_vertices=True, style="wireframe")
            else:
                plt.add_mesh(mesh, scalars="edge_colors", show_scalar_bar=False, point_size=node_point_size, line_width=mesh_line_width,
                             smooth_shading=True, clim=Color.color_limits(), show_vertices=True, style="wireframe")

        if show_edge_weights:
            all_edges = reconvert_faces(mesh.lines)
            edge_weights = {pos: str(weight) for pos, weight in enumerate(mesh.cell_data['edge_weight'])}
            edge_labels = []
            edge_points = []
            for edge_pos, label in edge_weights.items():
                point1 = mesh.points[all_edges[edge_pos][0]]
                point2 = mesh.points[all_edges[edge_pos][1]]
                center = (point1 + point2) / 2
                edge_points.append(center)
                edge_labels.append(label)
            plt.add_point_labels(edge_points, edge_labels, show_points=False, always_visible=True)

        plt.add_points(mesh.points, scalars=mesh["colors"], point_size=node_point_size, show_scalar_bar=False,
                      clim=Color.color_limits())
        plt.enable_anti_aliasing('msaa', multi_samples=16)

        plt.camera_position = camera_position
        if filename is None:
            if print_camera_position:
                plt.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, lambda *args: print(str(plt.camera_position)))
            plt.show(title=window_title)
        else:
            plt.screenshot(filename=str(filename), scale=5)
