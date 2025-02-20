"""Plotting dual lattice & constructed primary lattice from graph definition."""
import abc
import collections
import dataclasses
import itertools
import pathlib
import re
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import ClassVar, Optional

import numpy as np
import numpy.linalg
import numpy.typing as npt
import pyvista
import pyvista.plotting
import pyvista.plotting.themes
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from scipy.spatial import Delaunay
import vtk

from framework.base import DualGraphNode, GraphNode, GraphEdge
from framework.stabilizers import Color
from framework.util import compute_simplexes

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

    for points in [plane1, plane2, plane3]:
        points = np.asarray(points)
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
    for i in [0, 1]:
        a.append([line1[0][i] - line1[1][i], line2[0][i] - line2[1][i]])
        b.append(line1[0][i] + line2[0][i])
    # a = [line1[0], line2[0]]
    # b = [line1[0].dot(line1[1]), line2[0].dot(line2[1])]
    # print(a)
    # print(b)
    s = np.linalg.lstsq(a, b)
    #print(line1)
    #print(line2)
    #print(line1[0] + s[0][0] * (line1[0] - line1[1]))
    #exit()
    return line1[0] + s[0][0] * (line1[0] - line1[1])


def distance_to_plane(plane: list[np.ndarray], points: list[list[np.ndarray]]) -> list[float]:
    n = np.cross(plane[0] - plane[1], plane[0] - plane[2])
    n_normalized = n / np.linalg.norm(n)
    return [np.abs(np.dot(n_normalized, point - plane[0])) for point in points]


def distance_between_points(point1: list[np.ndarray], point2: list[np.ndarray]) -> float:
    return np.sqrt(sum((a-b)**2 for a, b in zip(point1, point2)))


@dataclasses.dataclass
class Plotter3D(abc.ABC):
    dual_graph: rx.PyGraph
    distance: int
    _dual_mesh: pyvista.PolyData = dataclasses.field(default=None, init=False)
    storage_dir: pathlib.Path = dataclasses.field(default=pathlib.Path(__file__).parent.parent.absolute())
    pyvista_theme: pyvista.plotting.themes.DocumentTheme = dataclasses.field(default=None, init=False)
    highes_id: int = dataclasses.field(default=0, init=False)
    _dualgraph_to_dualmesh: dict[int, int] = dataclasses.field(default=None, init=False)
    _dualmesh_to_dualgraph: dict[int, int] = dataclasses.field(default=None, init=False)
    dimension: ClassVar[int] = 3

    def __post_init__(self):
        self.pyvista_theme = self.get_plotting_theme()

    @property
    def dual_mesh(self) -> pyvista.PolyData:
        if not self._dual_mesh:
            self._dual_mesh = self._construct_dual_mesh()
        return self._dual_mesh

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
            faces = [[rustworkx2pyvista[index] for index in combination] for simplex in simplexes for combination in itertools.combinations(simplex, 3)]
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
        mandatory_qubits: set[int] = None
    ) -> pyvista.PolyData:
        """Create a 3D mesh of the given rustworkx Graph.

        Nodes must be GraphNode and edges GraphEdge objects.

        :param coordinates: Mapping of node indices of graph to 3D coordinates. Use them instead of calculating them.
        :param use_edges_colors: If true, use the color of the GraphEdge object instead of default colors.
        :param highlighted_nodes: Change color of given nodes to highlighted color. Take care to adjust the cmap of
            pyvista_theme to 'Color.highlighted_color_map' (otherwise there will be no visible effect).
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
        ret.cell_data["edge_index"] = [edge.index for edge in graph.edges() if include_edges_between_boundaries or not edge.is_edge_between_boundaries]
        ret.cell_data["edge_weight"] = [np.round(getattr(edge, "weight", -1), decimals=3) for edge in graph.edges() if include_edges_between_boundaries or not edge.is_edge_between_boundaries]

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

    def layout_primary_nodes(self, given_qubit_coordinates: dict[int, npt.NDArray[np.float64]]) -> (list[npt.NDArray[np.float64]], dict[int, int]):
        # volumes -> vertices

        points: list[npt.NDArray[np.float64]] = []
        # map each qubit to the position of its points coordinates
        qubit_to_pointpos: dict[int, int] = {}
        # map each qubit to the dual graph boundary nodes its tetrahedron touches
        qubit_to_boundaries: dict[int, list[DualGraphNode]] = dict()

        # group dual lattice cells (of the tetrahedron) by qubit
        qubit_to_facepositions: dict[int, list[int]] = defaultdict(list)
        for position, qubit in enumerate(self.dual_mesh.cell_data['qubits']):
            qubit_to_facepositions[qubit].append(position)

        dual_mesh_faces = reconvert_faces(self.dual_mesh.faces)
        for pointposition, (qubit, facepositions) in enumerate(qubit_to_facepositions.items()):
            dg_nodes = [self.get_dual_node(point_index) for point_index in
                        sorted(set().union(*[dual_mesh_faces[face_index] for face_index in facepositions]))]
            if given_qubit_coordinates:
                points.append(given_qubit_coordinates[qubit])
            else:
                tetrahedron = self.dual_mesh.extract_cells(facepositions)
                # find center of mass of the tetrahedron
                center = np.asarray([0.0, 0.0, 0.0])
                for point in tetrahedron.points:
                    center += point
                center = center / len(tetrahedron.points)
                points.append(center)
            qubit_to_pointpos[qubit] = pointposition
            qubit_to_boundaries[qubit] = [node for node in dg_nodes if node.is_boundary]

        points = self.postprocess_primary_node_layout(points, qubit_to_pointpos, qubit_to_boundaries)
        return points, qubit_to_pointpos

    @abc.abstractmethod
    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int],
            qubit_to_boundaries: dict[int, list[DualGraphNode]]
    ) -> list[npt.NDArray[np.float64]]:
        ...

    def construct_primary_mesh(self, highlighted_volumes: list[DualGraphNode] = None,
                               qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                               face_color: Color | list[Color] = None, node_color: Color | list[Color] = None,
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
        if face_color is not None and node_color is not None:
            raise ValueError("Only one of face_color and node_color may be present.")
        if face_color is not None and not isinstance(face_color, list):
            face_color = [face_color]
        if node_color is not None and not isinstance(node_color, list):
            node_color = [node_color]
        highlighted_volumes = highlighted_volumes or []

        points, qubit_to_pointpos = self.layout_primary_nodes(qubit_coordinates)
        qubits = list(qubit_to_pointpos.keys())
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}

        mandatory_face_qubits = mandatory_face_qubits or set(qubits)
        mandatory_cell_qubits = mandatory_cell_qubits or set(qubits)
        if face_syndrome_qubits:
            mandatory_face_qubits = face_syndrome_qubits

        # vertices -> volumes
        volumes: list[list[int]] = []
        volumes_by_pos: dict[int, set[int]] = {}
        volume_ids: list[int] = []
        volume_indices: list[int] = []
        volume_colors: list[Color] = []
        volume_colors_by_pos: dict[int, Color] = {}
        all_edges = set()
        present_edges = set()
        for pos, node in enumerate(self.dual_graph.nodes()):
            # do not add a volume for boundaries
            if node.is_boundary:
                # add pseudo volume_by_pos and volume_colors_by_pos to color edges
                volumes_by_pos[pos] = {qubit_to_pointpos[qubit] for qubit in node.qubits}
                volume_colors_by_pos[pos] = node.color
                continue
            add_volume = True
            if node.title and (match := re.match(r"\((?P<col>-?\d+),(?P<row>-?\d+),(?P<layer>-?\d+)\)", node.title)):
                layer, row, col = int(match.group("layer")), int(match.group("row")), int(match.group("col"))
                if lowest_title and not (lowest_title[0] <= col and lowest_title[1] <= row and lowest_title[2] <= layer ):
                    add_volume = False
                if highest_title and not (col <= highest_title[0] and row <= highest_title[1] and layer <= highest_title[2]):
                    add_volume = False
            if not mandatory_cell_qubits & set(node.qubits):
                add_volume = False
            # we only need the further stuff for color_edges
            if not add_volume and not color_edges:
                continue
            # each dual graph edge corresponds to a primary graph face
            all_face_qubits = [(edge.color, edge.qubits) for _, _, edge in self.dual_graph.out_edges(node.index)
                               if set(edge.qubits) & mandatory_face_qubits
                               and (face_syndrome_qubits is None or len(set(edge.qubits) & face_syndrome_qubits)) % 2 == 1]
            faces: list[list[int]] = []
            face_colors = []
            for f_color, face_qubits in all_face_qubits:
                face_points = [qubit_to_point[qubit] for qubit in face_qubits]
                # project the points to a 2D plane with 2D coordinates, then calculate their triangulation
                triangulation = Delaunay(project_to_2d_plane(face_points), qhull_options="QJ")
                # extract faces of the triangulation, take care to use the qubits
                tmp_point_map = {k: v for k, v in zip(range(triangulation.npoints), face_qubits)}
                simplexes = [[tmp_point_map[point] for point in face] for face in triangulation.simplices]
                face = [qubit_to_pointpos[qubit] for qubit in triangles_to_face(simplexes)]
                # otherwise, most faces would be included twice, once per volume
                if mandatory_face_qubits != set(qubits) and face in volumes:
                    continue
                if (face_color is None and node_color is None) or (node_color is not None and node.color in node_color):
                    if string_operator_qubits is None or string_operator_qubits & set(node.qubits):
                        faces.append(face)
                        if node in highlighted_volumes:
                            face_colors.append(node.color.highlight)
                        else:
                            face_colors.append(node.color)
                elif (face_color is not None and f_color in face_color):
                    # otherwise, most faces would be included twice, once per volume
                    if face in volumes:
                        continue
                    faces.append(face)
                    if node in highlighted_volumes:
                        face_colors.append(f_color.highlight)
                    else:
                        face_colors.append(f_color)
                if add_volume:
                    # 0, 1, 2, 3
                    # 3, 0, 1, 2
                    for edge in zip(face, [face[-1]] + face[:-1]):
                        all_edges.add(edge)
                        all_edges.add(edge[::-1])
                        if ((face_color is None and node_color is None)
                                or (face_color is not None and f_color in face_color)
                                or (node_color is not None and node.color in node_color)):
                            present_edges.add(edge)
                            present_edges.add(edge[::-1])
            # add volume faces
            if add_volume:
                volumes.extend(faces)
                # add volume ids
                volume_ids.extend([self.next_id] * len(faces))
                # add volume colors
                volume_colors.extend(face_colors)
                # save index of dual graph node
                volume_indices.extend([node.index] * len(faces))
            # needed for color_edges, will not be returned
            volumes_by_pos[pos] = set(itertools.chain.from_iterable(faces))
            volume_colors_by_pos[pos] = face_colors[0] if face_colors else -1
        present_point_pos = set(itertools.chain.from_iterable(present_edges))
        # only those qubits may support additional edges
        if string_operator_qubits:
            present_point_pos = {qubit_to_pointpos[qubit] for qubit in string_operator_qubits}
        lines: list[tuple[int, int]] = []
        line_ids = []
        line_colors = []
        if node_color is not None:
            for edge in sorted(all_edges - present_edges):
                if edge[0] in present_point_pos and edge[1] in present_point_pos:
                    lines.append(edge)
                    line_ids.append(self.next_id)
                    line_colors.append(node_color[0])
        elif color_edges:
            for pos1, pos2 in itertools.combinations(volumes_by_pos.keys(), 2):
                if volume_colors_by_pos[pos1] != volume_colors_by_pos[pos2] or volume_colors_by_pos[pos1] < 0:
                    continue
                edges = [edge for edge in itertools.product(volumes_by_pos[pos1], volumes_by_pos[pos2]) if edge in all_edges]
                if not edges:
                    continue
                # the distance check is a nasty ducktape fix: some of the faces (with 6 points and 2 connected squares)
                # do not produce meaningful delauny triangulations, i.e. one of the points is inside the triangulation
                # instead of at the boundary. This leads to edges between non-neighbour qubits, which is only relevant
                # if we color the edges here. One should fix this properly though...
                distances = [distance_between_points(points[edge[0]], points[edge[1]]) for edge in edges]
                min_distance = min(distances)
                for edge, distance in zip(edges, distances):
                    if edge not in lines and (edge[1], edge[0]) not in lines and 0.9*distance <= min_distance:
                        lines.append(edge)
                        line_ids.append(self.next_id)
                        line_colors.append(volume_colors_by_pos[pos1].highlight)
        ret = pyvista.PolyData(points, faces=convert_faces(volumes), lines=convert_faces(lines) if len(lines) else None)
        ret.point_data['qubits'] = qubits
        ret.cell_data['face_ids'] = [*line_ids, *volume_ids]
        ret.cell_data['colors'] = [*line_colors, *volume_colors]
        ret.cell_data['pyvista_indices'] = [*([-1]*len(lines)), *volume_indices]
        return ret

    @staticmethod
    @abc.abstractmethod
    def _layout_dual_nodes_factor(distance: int) -> Optional[float]:
        ...

    def layout_dual_nodes(self, primary_mesh: pyvista.PolyData) -> dict[int, npt.NDArray[np.float64]]:
        # compute the center of each volume
        ret: dict[int, npt.NDArray[np.float64]] = self.preprocess_dual_node_layout(primary_mesh)

        # calculate the center of all nodes
        center = np.asarray([0.0, 0.0, 0.0])
        for point in ret.values():
            center += point
        center = center / len(ret)

        # compute the position of each boundary node
        boundary_nodes = [node for node in self.dual_graph.nodes() if node.is_boundary]
        for node in boundary_nodes:
            adjacent_nodes = [index for index in self.dual_graph.neighbors(node.index) if not self.dual_graph[index].is_boundary]
            face_center = np.asarray([0.0, 0.0, 0.0])
            for index, position in ret.items():
                if index not in adjacent_nodes:
                    continue
                face_center += position
            face_center = face_center / len(adjacent_nodes)

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
        camera_position: list[tuple[int, int, int]] = None,
        print_camera_position: bool = False,
        filename: pathlib.Path = None,
    ) -> None:
        # use default values
        if point_size is None:
            point_size = 15 if filename is None else 120
        if line_width is None:
            line_width = 1 if filename is None else 10

        plt = pyvista.plotting.Plotter(theme=self.pyvista_theme, off_screen=filename is not None)
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

        plt.camera_position = camera_position
        if filename is None:
            if print_camera_position:
                plt.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, lambda *args: print(str(plt.camera_position)))
            plt.show()
        else:
            plt.screenshot(filename=str(filename), scale=5)
        return None

    def plot_primary_mesh(
        self,
        *,
        show_qubit_labels: bool = False,
        point_size: int = None,
        highlighted_volumes: list[DualGraphNode] = None,
        highlighted_qubits: list[int] = None,
        qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
        only_faces_with_color: Color | list[Color] = None,
        only_nodes_with_color: Color | list[Color] = None,
        lowest_title: tuple[int, int, int] = None,
        highest_title: tuple[int, int, int] = None,
        mandatory_face_qubits: set[int] = None,
        string_operator_qubits: set[int] = None,
        color_edges: bool = False,
        show_normal_qubits: bool = True,
        line_width: float = 3,
        transparent_faces: bool = False,
        mandatory_cell_qubits: set[int] = None,
        face_syndrome_qubits: set[int] = None,
        camera_position: list[tuple[int, int, int]] = None,
        print_camera_position: bool = False,
        filename: pathlib.Path = None,
    ) -> None:
        # set default values
        if point_size is None:
            point_size = 15 if filename is None else 70

        plt = self._plot_primary_mesh_internal(
            show_qubit_labels=show_qubit_labels,
            point_size=point_size,
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
            plt.show()
        else:
            plt.screenshot(filename=str(filename), scale=5)
        return None

    def _plot_primary_mesh_internal(
        self,
        *,
        show_qubit_labels: bool = False,
        point_size: int = None,
        highlighted_volumes: list[DualGraphNode] = None,
        highlighted_qubits: list[int] = None,
        qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
        only_faces_with_color: Color | list[Color] = None,
        only_nodes_with_color: Color | list[Color] = None,
        lowest_title: tuple[int, int, int] = None,
        highest_title: tuple[int, int, int] = None,
        mandatory_face_qubits: set[int] = None,
        string_operator_qubits: set[int] = None,
        color_edges: bool = False,
        show_normal_qubits: bool = True,
        line_width: float = 3,
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

        if self.dimension == 3 and not transparent_faces:
            theme = self.get_plotting_theme()
            theme.cmap = Color.highlighted_color_map_3d()
        else:
            theme = self.pyvista_theme

        plt = pyvista.plotting.Plotter(theme=theme, off_screen=off_screen)
        # extract lines from mesh, plot them separately
        if only_nodes_with_color is not None or color_edges:
            line_poses = reconvert_faces(mesh.lines)
            edge_mesh = pyvista.PolyData(mesh.points, lines=mesh.lines)
            if only_nodes_with_color is not None:
                line_color = Color.highlighted_color_map_3d().colors[Color(only_nodes_with_color).highlight]
                plt.add_mesh(edge_mesh, show_scalar_bar=False, line_width=25, smooth_shading=True,
                             color=line_color, point_size=point_size, show_vertices=False, style="wireframe")
            elif color_edges:
                if line_width is None:
                    raise ValueError(line_width)
                edge_mesh.cell_data["colors"] = list(mesh.cell_data["colors"])[:len(line_poses)]
                plt.add_mesh(edge_mesh, show_scalar_bar=False, line_width=line_width, smooth_shading=True,
                             clim=Color.color_limits(), scalars="colors", point_size=point_size, show_vertices=False, style="wireframe")
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
        for qubits, color in [(normal_qubits, "indigo"), (highlighted_qubits, "violet")]:
            positions = [pos for pos, qubit in enumerate(mesh.point_data['qubits']) if qubit in qubits and (pos in used_qubit_pos or face_syndrome_qubits)]
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
        only_faces_with_color: Color | list[Color] = None,
        only_nodes_with_color: Color | list[Color] = None,
        lowest_title: tuple[int, int, int] = None,
        highest_title: tuple[int, int, int] = None,
        mandatory_face_qubits: set[int] = None,
        string_operator_qubits: set[int] = None,
        color_edges: bool = False,
        show_normal_qubits: bool = True,
        transparent_faces: bool = False,
        highlighted_edges: list[GraphEdge] = None,
        qubit_point_size: int = None,
        mesh_line_width: int = None,
        node_point_size: int = None,
        show_normal_edges: bool = True,
        primary_line_width: int = None,
        highlighted_line_width: int = None,
        mesh_line_color: Optional[str] = None,
        mandatory_cell_qubits: set[int] = None,
        face_syndrome_qubits: set[int] = None,
        show_edge_weights: bool = False,
        camera_position: list[tuple[int, int, int]] = None,
        print_camera_position: bool = False,
        filename: pathlib.Path = None,
    ) -> None:
        """Return the plotter preloaded with the debug and primary mesh."""
        # use default values
        if qubit_point_size is None:
            qubit_point_size = 15 if filename is None else 70
        if node_point_size is None:
            node_point_size = 20 if filename is None else 120
        if mesh_line_width is None:
            mesh_line_width = 1 if filename is None else 10
        if primary_line_width is None:
            primary_line_width = 1 if filename is None else 10
        if highlighted_line_width is None:
            highlighted_line_width = 2 if filename is None else 10

        plt = self._plot_primary_mesh_internal(
            show_qubit_labels=show_qubit_labels,
            point_size=qubit_point_size,
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
                normal_edge = pyvista.PolyData(mesh.points, lines=convert_faces([edge for pos, edge in enumerate(all_edges) if pos in normal_edge_pos]))
                plt.add_mesh(normal_edge, show_scalar_bar=False, line_width=mesh_line_width, smooth_shading=True, color="silver",
                             point_size=node_point_size, show_vertices=True, style="wireframe")
            highlighted_edge_pos = [pos for pos, index in enumerate(mesh.cell_data['edge_index']) if index in highlighted_edge_indices]
            highlighted_edge = pyvista.PolyData(mesh.points, lines=convert_faces([edge for pos, edge in enumerate(all_edges) if pos in highlighted_edge_pos]))
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
            plt.show()
        else:
            plt.screenshot(filename=str(filename), scale=5)
        return None



def rotate_points(points: list[npt.NDArray[np.float64]], x_angle: float, y_angle: float, z_angle: float, center: npt.NDArray[np.float64]=None):
    """Rotate all points by x, y and then z axis around the given center."""
    if center is not None:
        points = [point - center for point in points]

    x_mat = np.array([[1, 0, 0],
                      [0, np.cos(x_angle), -np.sin(x_angle)],
                      [0, np.sin(x_angle), np.cos(x_angle)]])
    points = [x_mat.dot(point.transpose()) for point in points]
    y_mat = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                      [0, 1, 0],
                      [-np.sin(y_angle), 0, np.cos(y_angle)]])
    points = [y_mat.dot(point.transpose()) for point in points]
    z_mat = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                      [np.sin(z_angle), np.cos(z_angle), 0],
                      [0, 0, 1]])
    points = [z_mat.dot(point.transpose()) for point in points]

    if center is not None:
        points = [point + center for point in points]
    return points


@dataclasses.dataclass
class TetrahedronPlotter(Plotter3D):
    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int],
            qubit_to_boundaries: dict[int, list[DualGraphNode]]
    ) -> list[npt.NDArray[np.float64]]:
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}
        corner_qubits = {qubit: nodes for qubit, nodes in qubit_to_boundaries.items() if len(nodes) == 3}
        border_qubits = {qubit: nodes for qubit, nodes in qubit_to_boundaries.items() if len(nodes) == 2}
        border_to_qubit: dict[tuple[int, int]: list[int]] = collections.defaultdict(list)
        for qubit, nodes in border_qubits.items():
            border_to_qubit[tuple(sorted([nodes[0].index, nodes[1].index]))].append(qubit)
        boundary_qubits = {qubit: nodes for qubit, nodes in qubit_to_boundaries.items() if len(nodes) == 1}
        boundary_to_qubit: dict[int, list[int]] = collections.defaultdict(list)
        for qubit, nodes in boundary_qubits.items():
            boundary_to_qubit[nodes[0].index].append(qubit)
        bulk_qubits = {qubit for qubit, nodes in qubit_to_boundaries.items() if len(nodes) == 0}


        # TODO use enhanced method for d=5
        if self.distance <= 5:
            # move corner qubits more outward
            dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
            for qubit in corner_qubits:
                dual_mesh_center += qubit_to_point[qubit]
            dual_mesh_center /= len(corner_qubits)
            for qubit in corner_qubits:
                coordinate = qubit_to_point[qubit] + 1.25 * (qubit_to_point[qubit] - dual_mesh_center)
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate
        else:
            # TODO fix placement of border qubits
            # determine position of corner qubits from boundary planes
            boundary_to_qubit_point: dict[int, list[np.ndarray]] = {}
            for boundary, qubits in boundary_to_qubit.items():
                boundary_to_qubit_point[boundary] = project_to_3d_plane([qubit_to_point[qubit] for qubit in qubits])
                for qubit, coordinate in zip(qubits, boundary_to_qubit_point[boundary]):
                    qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate
            for boundary1, boundary2, boundary3 in itertools.combinations(boundary_to_qubit.keys(), 3):
                coordinate = cross_point_3_planes(boundary_to_qubit_point[boundary1], boundary_to_qubit_point[boundary2],
                                                  boundary_to_qubit_point[boundary3])
                qubit = (set(self.dual_graph[boundary1].qubits) & set(self.dual_graph[boundary2].qubits) & set(
                    self.dual_graph[boundary3].qubits)).pop()
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

            # move corner qubits more inward
            dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
            for qubit in corner_qubits:
                dual_mesh_center += qubit_to_point[qubit]
            dual_mesh_center /= len(corner_qubits)
            for qubit in corner_qubits:
                coordinate = qubit_to_point[qubit] - 0.2 * (qubit_to_point[qubit] - dual_mesh_center)
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

        # move border qubits to the line spanned by their respective corner qubits, equally spaced
        for boundary_indices, qubits in border_to_qubit.items():
            boundary1 = self.dual_graph[boundary_indices[0]]
            boundary2 = self.dual_graph[boundary_indices[1]]

            # project qubits to the border, sort them by in order of appearance
            line_qubits = list(set(boundary1.qubits) & set(boundary2.qubits) & set(corner_qubits))
            if len(line_qubits) != 2:
                raise ValueError
            line = [qubit_to_point[line_qubits[0]], qubit_to_point[line_qubits[1]]]
            tmp = {qubit: distance_between_points(line[0], project_to_line(line, qubit_to_point[qubit])) for qubit in qubits}
            qubit_order = sorted(tmp, key=lambda x: tmp[x])

            # assign equal-spaced coordinates to each qubit
            qubit_distance = (line[1] - line[0]) / (len(qubits) + 1)
            for i, qubit in enumerate(qubit_order, start=1):
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = line[0] + i*qubit_distance

        # project boundary qubits to the plane spanned by their corner qubits
        for qubit, boundaries in boundary_qubits.items():
            plane_qubits = list(set(boundaries[0].qubits) & set(corner_qubits))
            if len(plane_qubits) != 3:
                raise ValueError
            [new_coordinate] = project_to_given_plane([qubit_to_point[plane_qubits[0]], qubit_to_point[plane_qubits[1]],
                                                       qubit_to_point[plane_qubits[2]]], [qubit_to_point[qubit]])
            qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = new_coordinate

        # move bulk qubits more to center
        dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
        for qubit in corner_qubits:
            dual_mesh_center += qubit_to_point[qubit]
        dual_mesh_center /= len(corner_qubits)
        for qubit in bulk_qubits:
            coordinate = qubit_to_point[qubit] - 0.25 * (qubit_to_point[qubit] - dual_mesh_center)
            qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

        # center the qubits around (0,0,0)
        dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
        for qubit in corner_qubits:
            dual_mesh_center += qubit_to_point[qubit]
        dual_mesh_center /= len(corner_qubits)
        for qubit, coordinate in qubit_to_point.items():
            qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate - dual_mesh_center

        return points

    @staticmethod
    def _layout_dual_nodes_factor(distance: int) -> Optional[float]:
        return {
            3: 4.5,
            5: 3,
        }.get(distance)

    def preprocess_dual_node_layout(self, primary_mesh: pyvista.PolyData):
        lines = reconvert_faces(primary_mesh.lines)
        faces = reconvert_faces(primary_mesh.faces)
        dg_index_to_points: dict[int, dict[int, npt.NDArray[np.float64]]] = defaultdict(dict)
        for dg_index, face in zip(primary_mesh.cell_data['pyvista_indices'][len(lines):], faces):
            for pos in face:
                dg_index_to_points[dg_index][primary_mesh.point_data['qubits'][pos]] = primary_mesh.points[pos]

        boundary_nodes = [node for node in self.dual_graph.nodes() if node.is_boundary]
        corner_qubits = {(set(node1.qubits) & set(node2.qubits) & set(node3.qubits)).pop()
                         for node1, node2, node3 in itertools.combinations(boundary_nodes, 3)}

        # compute the center of each volume
        ret: dict[int, npt.NDArray[np.float64]] = {}
        for dg_index, points in dg_index_to_points.items():
            center = np.asarray([0.0, 0.0, 0.0])
            divisor = len(points)
            for qubit, point in points.items():
                center += point
                if qubit in corner_qubits:
                    center += 5 * point
                    divisor += 5
            center = center / divisor
            ret[dg_index] = center

        return ret


@dataclasses.dataclass
class CubicPlotter(Plotter3D):
    def _primary_boundary_edge_factor(self, coordinate, boundary_edge_center) -> float:
        if self.distance == 4:
            factor = 0.55
        elif self.distance == 6:
            factor = 0.5
            if distance_between_points(coordinate, boundary_edge_center) < 50:
                factor = 0.2
        else:
            factor = 0.5
        return factor

    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int],
            qubit_to_boundaries: dict[int, list[DualGraphNode]]
    ) -> list[npt.NDArray[np.float64]]:
        """Move qubits at the outside more outward, to form an even plane at each boundary."""
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}
        boundary_nodes = [node for node in self.dual_graph.nodes() if node.is_boundary]
        corner_qubits = {qubit for qubit, nodes in qubit_to_boundaries.items() if len(nodes) == 3}
        border_qubits = {qubit: nodes for qubit, nodes in qubit_to_boundaries.items() if len(nodes) == 2}

        # calculate the reference planes
        boundary_to_reference_plane: dict[int, list[np.ndarray]] = {}
        for node in boundary_nodes:
            face_corner_qubit_coordinates = [qubit_to_point[qubit] for qubit in set(node.qubits) & corner_qubits]
            max_distance = 0
            reference_plane = []
            for neighbour_index in self.dual_graph.neighbors(node.index):
                neighbour = self.dual_graph[neighbour_index]
                if neighbour.is_boundary:
                    continue
                references = [qubit_to_point[qubit] for qubit in set(node.qubits) & set(neighbour.qubits)]
                if all(distance > max_distance for distance in
                       distance_to_plane(face_corner_qubit_coordinates, references)):
                    reference_plane = references
                    max_distance = min(distance_to_plane(face_corner_qubit_coordinates, references))
            boundary_to_reference_plane[node.index] = reference_plane

        # move all points at a boundary to the respective plain.
        for node_index, reference_plane in boundary_to_reference_plane.items():
            if not reference_plane:
                continue
            node = self.dual_graph[node_index]
            face_qubit_coordinates = [qubit_to_point[qubit] for qubit in node.qubits]
            plane_face_qubit_coordinates = project_to_given_plane(reference_plane, face_qubit_coordinates)

            plane_center = np.asarray([0.0, 0.0, 0.0])
            for point in plane_face_qubit_coordinates:
                plane_center += point
            plane_center = plane_center / len(plane_face_qubit_coordinates)

            for qubit, coordinate in zip(node.qubits, plane_face_qubit_coordinates):
                # move all qubits which are not on a boundary edge away from the center
                if (qubit not in corner_qubits and qubit not in border_qubits
                        and distance_between_points(coordinate, plane_center) > 30):
                    coordinate = coordinate + 0.4 * (coordinate - plane_center)
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

            for qubit in set(border_qubits) & set(node.qubits):
                coordinate = qubit_to_point[qubit]
                boundary_node1 = border_qubits[qubit][0]
                boundary_node2 = border_qubits[qubit][1]
                relevant_corner_qubits = list(corner_qubits & set(boundary_node1.qubits) & set(boundary_node2.qubits))
                if len(relevant_corner_qubits) != 2:
                    raise RuntimeError(relevant_corner_qubits)
                corner_qubit1 = qubit_to_point[relevant_corner_qubits[0]]
                corner_qubit2 = qubit_to_point[relevant_corner_qubits[1]]
                boundary_edge_center = (corner_qubit1 + corner_qubit2) / 2
                factor = self._primary_boundary_edge_factor(coordinate, boundary_edge_center)
                coordinate = coordinate + factor * (coordinate - boundary_edge_center)
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

        return points

    @staticmethod
    def _layout_dual_nodes_factor(distance: int) -> Optional[float]:
        return {
            4: 1.5,
            6: 1.0,
        }.get(distance)

    def preprocess_dual_node_layout(self, primary_mesh: pyvista.PolyData):
        lines = reconvert_faces(primary_mesh.lines)
        faces = reconvert_faces(primary_mesh.faces)
        dg_index_to_points: dict[int, dict[int, npt.NDArray[np.float64]]] = defaultdict(dict)
        for dg_index, face in zip(primary_mesh.cell_data['pyvista_indices'][len(lines):], faces):
            for pos in face:
                dg_index_to_points[dg_index][primary_mesh.point_data['qubits'][pos]] = primary_mesh.points[pos]

        boundary_nodes = [node for node in self.dual_graph.nodes() if node.is_boundary]
        corner_qubits = set()
        for node1, node2, node3 in itertools.combinations(boundary_nodes, 3):
            corner_qubits.update(set(node1.qubits) & set(node2.qubits) & set(node3.qubits))
        border_qubits = set()
        for node1, node2 in itertools.combinations(boundary_nodes, 2):
            border_qubits.update(set(node1.qubits) & set(node2.qubits))
        border_qubits -= corner_qubits

        # compute the center of each volume
        ret: dict[int, npt.NDArray[np.float64]] = {}
        for dg_index, points in dg_index_to_points.items():
            center = np.asarray([0.0, 0.0, 0.0])
            divisor = len(points)
            for qubit, point in points.items():
                center += point
                # align the center of the truncated chamfered cubes correctly
                if qubit in border_qubits and len(points) == 22:
                    center += 8 * point
                    divisor += 8
            center = center / divisor
            ret[dg_index] = center

        return ret


@dataclasses.dataclass
class Plotter2D(Plotter3D):
    dimension: ClassVar[int] = 2

    @staticmethod
    def _primary_distance_to_boundarynodeoffset(distance: int) -> Optional[float]:
        return {
            4: 0.3,
            6: 0.37,
        }.get(distance)

    @staticmethod
    def _primary_corne_node_offset(distance: int) -> float:
        return {
            4: 0.2,
            6: 0.13,
        }.get(distance, 0.3)

    def _primary_boundary_edge_factor(self, coordinate, boundary_edge_center) -> float:
        if self.distance == 4:
            factor = 0.8
        elif self.distance == 6:
            factor = 0.6
            if distance_between_points(coordinate, boundary_edge_center) < 50:
                factor = 0.0
        else:
            factor = 0.5
        return factor

    def construct_primary_mesh(self, highlighted_volumes: list[DualGraphNode] = None,
                               qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                               face_color: Color | list[Color] = None, node_color: Color | list[Color] = None,
                               lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                               mandatory_face_qubits: set[int] = None, string_operator_qubits: set[int] = None,
                               color_edges: bool = False, transparent_faces: bool = False, mandatory_cell_qubits: set[int] = None):
        """Construct primary mesh from dual_mesh.

        :param qubit_coordinates: Use them instead of calculating the coordinates from the dual_mesh.
        """
        highlighted_faces = highlighted_volumes or []
        # group dual lattice cells (of the triangle) by qubit
        qubit_to_facepositions: dict[int, list[int]] = defaultdict(list)
        for position, qubit in enumerate(self.dual_mesh.cell_data['qubits']):
            qubit_to_facepositions[qubit].append(position)

        # volumes -> vertices
        points: list[npt.NDArray[np.float64]] = []
        qubit_to_point: dict[int, npt.NDArray[np.float64]] = {}
        qubit_to_pointposition: dict[int, int] = {}
        qubits: list[int] = []
        # map of qubits on boundary edge to the dg_node index of this boundary
        boundary_edge_qubits: dict[int, int] = {}
        corner_qubits: set[int] = set()
        dual_mesh_faces = reconvert_faces(self.dual_mesh.faces)
        # determine center of dual mesh, to translate corner qubits in relation to this
        dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
        for point in self.dual_mesh.points:
            dual_mesh_center += point
        dual_mesh_center /= len(self.dual_mesh.points)
        for pointposition, (qubit, facepositions) in enumerate(qubit_to_facepositions.items()):
            dg_nodes = [self.get_dual_node(point_index) for point_index in sorted(set().union(
                *[dual_mesh_faces[face_index] for face_index in facepositions]))]
            triangle = self.dual_mesh.extract_cells(facepositions)
            # find center of mass of the tetrahedron
            center = np.asarray([0.0, 0.0, 0.0])
            for point in triangle.points:
                center += point
            center = center / len(triangle.points)
            if sum(node.is_boundary for node in dg_nodes) == 1:
                node_indices = sorted(node.index for node in dg_nodes if node.is_boundary)
                boundary_edge_qubits[qubit] = node_indices[0]
            # exactly three nodes are boundary nodes
            if sum(node.is_boundary for node in dg_nodes) == 2:
                center += self._primary_corne_node_offset(self.distance) * (center - dual_mesh_center)
                corner_qubits.add(qubit)
            # use given coordinates if provided
            if qubit_coordinates:
                points.append(qubit_coordinates[qubit])
                qubit_to_point[qubit] = qubit_coordinates[qubit]
            else:
                points.append(center)
                qubit_to_point[qubit] = center
            qubit_to_pointposition[qubit] = pointposition
            qubits.append(qubit)

        for node in self.dual_graph.nodes():
            if not node.is_boundary:
                continue
            for qubit in set(boundary_edge_qubits) & set(node.qubits):
                coordinate = qubit_to_point[qubit]
                boundary_node = self.dual_graph[boundary_edge_qubits[qubit]]
                relevant_corner_qubits = list(corner_qubits & set(boundary_node.qubits))
                if len(relevant_corner_qubits) != 2:
                    raise RuntimeError(relevant_corner_qubits)
                corner_qubit1 = qubit_to_point[relevant_corner_qubits[0]]
                corner_qubit2 = qubit_to_point[relevant_corner_qubits[1]]
                boundary_edge_center = (corner_qubit1 + corner_qubit2) / 2
                coordinate = project_to_line([corner_qubit1, corner_qubit2], coordinate)
                factor = self._primary_boundary_edge_factor(coordinate, boundary_edge_center)
                coordinate = coordinate + factor * (coordinate - boundary_edge_center)
                qubit_to_point[qubit] = coordinate
                points[qubit_to_pointposition[qubit]] = coordinate

        # vertices -> faces
        faces: list[list[int]] = []
        faces_by_pos: dict[int, set[int]] = {}
        face_ids: list[int] = []
        face_indices: list[int] = []
        face_colors: list[Color] = []
        face_colors_by_pos: dict[int, Color] = {}
        all_edges: set[tuple[int, int]] = set()
        for pos, node in enumerate(self.dual_graph.nodes()):
            # do not add a triangle for boundaries
            if node.is_boundary:
                # add pseudo faces_by_pos and face_colors_by_pos to color edges
                faces_by_pos[pos] = {qubit_to_pointposition[qubit] for qubit in node.qubits}
                face_colors_by_pos[pos] = node.color
                continue
            # 2 dimensional coordinates are enough, z coordinate is always 0
            face_points = [(qubit_to_point[qubit][0], qubit_to_point[qubit][1]) for qubit in node.qubits]
            triangulation = Delaunay(face_points, qhull_options="QJ")
            # extract faces of the triangulation, take care to use the qubits
            tmp_point_map = {k: v for k, v in zip(range(triangulation.npoints), node.qubits)}
            simplexes = [[tmp_point_map[point] for point in face] for face in triangulation.simplices]
            face = [qubit_to_pointposition[qubit] for qubit in triangles_to_face(simplexes)]
            for edge in zip(face + [face[-1]], face[1:] + [face[0]]):
                all_edges.add(edge)
                all_edges.add((edge[1], edge[0]))
            faces.append(face)
            faces_by_pos[pos] = set(face)
            face_ids.append(self.next_id)
            face_indices.append(node.index)
            face_color = node.color.highlight if node in highlighted_faces else node.color
            face_colors.append(face_color)
            face_colors_by_pos[pos] = face_color

        lines: list[tuple[int, int]] = []
        line_ids = []
        line_colors = []
        if color_edges:
            for pos1, pos2 in itertools.combinations(faces_by_pos.keys(), 2):
                if face_colors_by_pos[pos1] != face_colors_by_pos[pos2]:
                    continue
                for edge in itertools.product(faces_by_pos[pos1], faces_by_pos[pos2]):
                    if edge in all_edges and edge not in lines and (edge[1], edge[0]) not in lines:
                        lines.append(edge)
                        line_ids.append(self.next_id)
                        line_colors.append(face_colors_by_pos[pos1].highlight)
        ret = pyvista.PolyData(points, faces=convert_faces(faces), lines=convert_faces(lines) if len(lines) else None)
        ret.point_data['qubits'] = qubits
        ret.cell_data['face_ids'] = [*line_ids, *face_ids]
        ret.cell_data['colors'] = [*line_colors, *face_colors]
        ret.cell_data['pyvista_indices'] = [*([-1]*len(lines)), *face_indices]
        return ret

    @staticmethod
    def _layout_dual_nodes_factor(distance: int) -> Optional[float]:
        return {
            4: 1.2,
            6: 0.96,
        }.get(distance)
