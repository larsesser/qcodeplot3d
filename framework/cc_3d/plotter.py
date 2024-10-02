"""Plotting dual lattice & constructed primary lattice from graph definition."""
import dataclasses
import itertools
import pathlib
import re
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import ClassVar, Optional

import numpy as np
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


def project_to_plane(points: list[list[float]]) -> list[list[float]]:
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


def distance_to_plane(plane: list[np.ndarray], points: list[list[np.ndarray]]) -> list[float]:
    n = np.cross(plane[0] - plane[1], plane[0] - plane[2])
    n_normalized = n / np.linalg.norm(n)
    return [np.abs(np.dot(n_normalized, point - plane[0])) for point in points]


def distance_between_points(point1: list[np.ndarray], point2: list[np.ndarray]) -> float:
    return np.sqrt(sum((a-b)**2 for a, b in zip(point1, point2)))


@dataclasses.dataclass
class Plotter3D:
    dual_graph: rx.PyGraph
    distance: int
    _dual_mesh: pyvista.PolyData = dataclasses.field(default=None, init=False)
    _primary_mesh: pyvista.PolyData = dataclasses.field(default=None, init=False)
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

    @property
    def primary_mesh(self) -> pyvista.PolyData:
        if not self._primary_mesh:
            self._primary_mesh = self._construct_primary_mesh()
        return self._primary_mesh

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

    def get_3d_coordinates(self, graph: rx.PyGraph) -> dict[int, npt.NDArray[np.float64]]:
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
        node2coordinates = self.get_3d_coordinates(self.dual_graph)
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

    def construct_debug_mesh(self, graph: rx.PyGraph, coordinates: dict[int, npt.NDArray[np.float64]] = None,
                             use_edges_colors: bool = False, edge_color: Color = None,
                             highlighted_nodes: list[GraphNode] = None, highlighted_edges: list[GraphEdge] = None,
                             include_edges_between_boundaries: bool = True) -> pyvista.PolyData:
        """Create a 3D mesh of the given rustworkx Graph.

        Nodes must be GraphNode and edges GraphEdge objects.

        :param coordinates: Mapping of node indices of graph to 3D coordinates. Use them instead of calculating them.
        :param use_edges_colors: If true, use the color of the GraphEdge object instead of default colors.
        :param highlighted_nodes: Change color of given nodes to highlighted color. Take care to adjust the cmap of
            pyvista_theme to 'Color.highlighted_color_map' (otherwise there will be no visible effect).
        """
        highlighted_nodes = highlighted_nodes or []
        highlighted_edges = highlighted_edges or []
        # calculate positions of points (or use given coordinates)
        node2coordinates = coordinates or self.get_3d_coordinates(graph)
        points = np.asarray([node2coordinates[index] for index in graph.node_indices()])

        # generate pyvista edges from rustworkx edges
        rustworkx2pyvista = {rustworkx_index: pyvista_index for pyvista_index, rustworkx_index in enumerate(graph.node_indices())}
        lines = [[rustworkx2pyvista[edge.node1.index], rustworkx2pyvista[edge.node2.index]] for edge in graph.edges()
                 if include_edges_between_boundaries or not edge.is_edge_between_boundaries]
        ret = pyvista.PolyData(points, lines=convert_faces(lines))

        # remember which nodes are boundary nodes
        boundaries = [node.is_boundary for node in graph.nodes()]
        ret["is_boundary"] = boundaries

        # remember the edge idex
        ret.cell_data["edge_index"] = [edge.index for edge in graph.edges() if include_edges_between_boundaries or not edge.is_edge_between_boundaries]

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
                if edge.color:
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

    def _construct_primary_mesh(self, highlighted_volumes: list[DualGraphNode] = None,
                                qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                                face_color: Color | list[Color] = None, node_color: Color | list[Color] = None,
                                lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                                mandatory_face_qubits: set[int] = None, string_operator_qubits: set[int] = None,
                                color_edges: bool = False, mandatory_cell_qubits: set[int] = None) -> pyvista.PolyData:
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
        # group dual lattice cells (of the tetrahedron) by qubit
        qubit_to_facepositions: dict[int, list[int]] = defaultdict(list)
        for position, qubit in enumerate(self.dual_mesh.cell_data['qubits']):
            qubit_to_facepositions[qubit].append(position)

        # volumes -> vertices
        points: list[npt.NDArray[np.float64]] = []
        qubit_to_point: dict[int, npt.NDArray[np.float64]] = {}
        qubit_to_pointposition: dict[int, int] = {}
        qubits: list[int] = []
        corner_qubits: set[int] = set()
        # map of qubits on boundary edge to the dg_node indicies of this boundary edge
        boundary_edge_qubits: dict[int, tuple[int, int]] = {}
        dual_mesh_faces = reconvert_faces(self.dual_mesh.faces)
        # determine center of dual mesh, to translate corner qubits in relation to this
        dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
        for point in self.dual_mesh.points:
            dual_mesh_center += point
        dual_mesh_center /= len(self.dual_mesh.points)
        for pointposition, (qubit, facepositions) in enumerate(qubit_to_facepositions.items()):
            dg_nodes = [self.get_dual_node(point_index) for point_index in sorted(set().union(
                *[dual_mesh_faces[face_index] for face_index in facepositions]))]
            tetrahedron = self.dual_mesh.extract_cells(facepositions)
            # find center of mass of the tetrahedron
            center = np.asarray([0.0, 0.0, 0.0])
            for point in tetrahedron.points:
                center += point
            center = center / len(tetrahedron.points)
            if sum(node.is_boundary for node in dg_nodes) == 2:
                node_indices = sorted(node.index for node in dg_nodes if node.is_boundary)
                boundary_edge_qubits[qubit] = tuple(node_indices)
            # exactly three nodes are boundary nodes
            if sum(node.is_boundary for node in dg_nodes) == 3:
                corner_qubits.add(qubit)
                if self.distance == 3:
                    center = center + 1 * (center - dual_mesh_center)
            # use given coordinates if provided
            if qubit_coordinates:
                points.append(qubit_coordinates[qubit])
                qubit_to_point[qubit] = qubit_coordinates[qubit]
            else:
                points.append(center)
                qubit_to_point[qubit] = center
            qubit_to_pointposition[qubit] = pointposition
            qubits.append(qubit)
        # move qubits at the outside more outward, to form an even plane at each boundary (tested for cubic color codes)
        # first, calculate the reference planes ...
        boundary_to_reference_plane: dict[int, list[np.ndarray]] = {}
        for node in self.dual_graph.nodes():
            if not node.is_boundary:
                continue
            face_corner_qubit_coordinates = [qubit_to_point[qubit] for qubit in set(node.qubits) & corner_qubits]
            if self.distance == 3:
                reference_plane = face_corner_qubit_coordinates
            else:
                max_distance = 0
                reference_plane = []
                for neighbour_index in self.dual_graph.neighbors(node.index):
                    neighbour = self.dual_graph[neighbour_index]
                    if neighbour.is_boundary:
                        continue
                    references = [qubit_to_point[qubit] for qubit in set(node.qubits) & set(neighbour.qubits)]
                    if all(distance > max_distance for distance in distance_to_plane(face_corner_qubit_coordinates, references)):
                        reference_plane = references
                        max_distance = min(distance_to_plane(face_corner_qubit_coordinates, references))
            boundary_to_reference_plane[node.index] = reference_plane
        # ... then, apply the moving to the plane ...
        for node_index, reference_plane in boundary_to_reference_plane.items():
            if reference_plane == []:
                continue
            node = self.dual_graph[node_index]
            face_qubit_coordinates = [qubit_to_point[qubit] for qubit in node.qubits]
            plane_face_qubit_coordinates = project_to_given_plane(reference_plane, face_qubit_coordinates)
            plane_center = np.asarray([0.0, 0.0, 0.0])
            for point in plane_face_qubit_coordinates:
                plane_center += point
            plane_center = plane_center / len(plane_face_qubit_coordinates)
            for qubit, coordinate in zip(node.qubits, plane_face_qubit_coordinates):
                # ... and move all qubits which are not on a boundary edge away from the center
                if qubit in corner_qubits:
                    pass
                elif qubit in boundary_edge_qubits:
                    pass
                elif distance_between_points(coordinate, plane_center) > 30:
                    coordinate = coordinate + 0.4 * (coordinate - plane_center)
                qubit_to_point[qubit] = coordinate
                points[qubit_to_pointposition[qubit]] = coordinate
            for qubit in set(boundary_edge_qubits) & set(node.qubits):
                coordinate = qubit_to_point[qubit]
                boundary_node1 = self.dual_graph[boundary_edge_qubits[qubit][0]]
                boundary_node2 = self.dual_graph[boundary_edge_qubits[qubit][1]]
                relevant_corner_qubits = list(corner_qubits & set(boundary_node1.qubits) & set(boundary_node2.qubits))
                if len(relevant_corner_qubits) != 2:
                    raise RuntimeError(relevant_corner_qubits)
                corner_qubit1 = qubit_to_point[relevant_corner_qubits[0]]
                corner_qubit2 = qubit_to_point[relevant_corner_qubits[1]]
                boundary_edge_center = (corner_qubit1 + corner_qubit2) / 2
                factor = self._primary_boundary_edge_factor(coordinate, boundary_edge_center)
                coordinate = coordinate + factor * (coordinate - boundary_edge_center)
                qubit_to_point[qubit] = coordinate
                points[qubit_to_pointposition[qubit]] = coordinate
        mandatory_face_qubits = mandatory_face_qubits or set(qubits)
        mandatory_cell_qubits = mandatory_cell_qubits or set(qubits)

        pointpos_to_point: dict[int, npt.NDArray[np.float64]] = {}
        for pointpos, point in enumerate(points):
            pointpos_to_point[pointpos] = point

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
                volumes_by_pos[pos] = {qubit_to_pointposition[qubit] for qubit in node.qubits}
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
                               if set(edge.qubits) & mandatory_face_qubits]
            faces: list[list[int]] = []
            face_colors = []
            for f_color, face_qubits in all_face_qubits:
                face_points = [qubit_to_point[qubit] for qubit in face_qubits]
                # project the points to a 2D plane with 2D coordinates, then calculate their triangulation
                triangulation = Delaunay(project_to_plane(face_points), qhull_options="QJ")
                # extract faces of the triangulation, take care to use the qubits
                tmp_point_map = {k: v for k, v in zip(range(triangulation.npoints), face_qubits)}
                simplexes = [[tmp_point_map[point] for point in face] for face in triangulation.simplices]
                face = [qubit_to_pointposition[qubit] for qubit in triangles_to_face(simplexes)]
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
            present_point_pos = {qubit_to_pointposition[qubit] for qubit in string_operator_qubits}
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
                distances = [distance_between_points(pointpos_to_point[edge[0]], pointpos_to_point[edge[1]]) for edge in edges]
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
    def _dual_mesh_coordinates_from_primary_mesh_boundary_factor(distance: int) -> Optional[float]:
        return {
            3: 4.5,
            4: 1.5,
            6: 1.0,
        }.get(distance)

    def get_dual_mesh_coordinates_from_primary_mesh(self, primary_mesh: pyvista.PolyData = None) -> dict[int, npt.NDArray[np.float64]]:
        primary_mesh = primary_mesh or self.primary_mesh
        lines = reconvert_faces(primary_mesh.lines)
        faces = reconvert_faces(primary_mesh.faces)
        boundary_nodes = [node for node in self.dual_graph.nodes() if node.is_boundary]
        pointpos_to_point: dict[int, npt.NDArray[np.float64]] = {pos: point for pos, point in enumerate(primary_mesh.points)}
        pointpos_to_qubit: dict[int, int] = {pos: qubit for pos, qubit in enumerate(primary_mesh.point_data['qubits'])}
        dg_index_to_points: dict[int, list[tuple[int, npt.NDArray[np.float64]]]] = defaultdict(list)
        for dg_index, face in zip(primary_mesh.cell_data['pyvista_indices'][len(lines):], faces):
            for pos in face:
                dg_index_to_points[dg_index].append((pointpos_to_qubit[pos], pointpos_to_point[pos]))
        corner_qubits = set()
        if self.distance == 3:
            corner_qubits = {(set(node1.qubits) & set(node2.qubits) & set(node3.qubits)).pop()
                             for node1, node2, node3 in itertools.combinations(boundary_nodes, 3)}
        ret: dict[int, npt.NDArray[np.float64]] = {}
        # first, compute the center of each volume
        for dg_index, points in dg_index_to_points.items():
            center = np.asarray([0.0, 0.0, 0.0])
            divisor = len(points)
            for qubit, point in points:
                center += point
                if qubit in corner_qubits:
                    center += 5*point
                    divisor += 5
            center = center / divisor
            ret[dg_index] = center
        # calculate the center of all nodes
        center = np.asarray([0.0, 0.0, 0.0])
        for point in ret.values():
            center += point
        center = center / len(ret)
        # then, compute the position of each boundary node
        for node in boundary_nodes:
            adjacent_nodes = [index for index in self.dual_graph.neighbors(node.index) if not self.dual_graph[index].is_boundary]
            face_center = np.asarray([0.0, 0.0, 0.0])
            for index, position in ret.items():
                if index not in adjacent_nodes:
                    continue
                face_center += position
            face_center /= len(adjacent_nodes)

            # extrapolate the position of the boundary node from the line through center and face_center
            factor = self._dual_mesh_coordinates_from_primary_mesh_boundary_factor(self.distance) or 1
            ret[node.index] = face_center + factor*(face_center - center)
        return ret


    def get_primary_graph_qubit_coordinates(self, primary_mesh: pyvista.PolyData = None) -> dict[int, npt.NDArray[np.float64]]:
        primary_mesh = primary_mesh or self.primary_mesh
        return {qubit: coordinate for qubit, coordinate in zip(primary_mesh.point_data['qubits'], primary_mesh.points)}


    @staticmethod
    def explode(mesh: pyvista.PolyData, factor=0.4) -> pyvista.UnstructuredGrid:
        # group cells by id
        cells = defaultdict(list)
        for cell, anid in enumerate(mesh.cell_data['face_ids']):
            cells[anid].append(cell)

        # extract each cell, translate it by its center of mass, and recombine
        ret_points = []
        ret_cells = []
        ret_celltypes = []
        ret_color = []
        ret_color_points = []
        ret_face_ids = []
        ret_point_labels = []
        ret_qubits = []
        ret_edge_index = []
        for data in cells.values():
            volume = mesh.extract_cells(data)
            # translate by center of mass
            center = np.asarray([0.0, 0.0, 0.0])
            for point in volume.points:
                center += point
            center = center / len(volume.points)
            volume.translate(factor * center, inplace=True)
            ret_points.extend(volume.points)
            num_points = len(ret_points)
            point_converter = {key: value for key, value in zip(range(volume.n_points), range(num_points - volume.n_points, num_points))}
            # convert the cell point positions
            ret_cells.extend(convert_faces([[point_converter[point] for point in cell] for cell in reconvert_faces(volume.cells)]))
            ret_celltypes.extend(volume.celltypes)
            if 'colors' in volume.cell_data:
                ret_color.extend(volume.cell_data['colors'])
            if 'colors' in volume.point_data:
                ret_color_points.extend(volume.point_data['colors'])
            if 'face_ids' in volume.cell_data:
                ret_face_ids.extend(volume.cell_data['face_ids'])
            if 'point_labels' in volume.point_data:
                ret_point_labels.extend(volume.point_data['point_labels'])
            if 'qubits' in volume.point_data:
                ret_qubits.extend(volume.point_data['qubits'])
            if 'edge_index' in volume.cell_data:
                ret_edge_index.extend(volume.cell_data['edge_index'])
        ret = pyvista.UnstructuredGrid(ret_cells, ret_celltypes, ret_points)
        if ret_color:
            ret.cell_data['colors'] = ret_color
        if ret_color_points:
            ret.point_data['colors'] = ret_color_points
        if ret_face_ids:
            ret.cell_data['face_ids'] = ret_face_ids
        if ret_point_labels:
            ret.point_data['point_labels'] = ret_point_labels
        if ret_qubits:
            ret.point_data['qubits'] = ret_qubits
        if ret_edge_index:
            ret.cell_data['edge_index'] = ret_edge_index
        return ret

    def show_dual_mesh(self, show_labels: bool = False, explode_factor: float = 0.0, exclude_boundaries: bool = False, print_cpos: bool = False) -> None:
        plotter = self.get_dual_mesh_plotter(show_labels, explode_factor, exclude_boundaries, off_screen=False)

        def my_cpos_callback(*args):
            # plotter.add_text(str(plotter.camera_position), name="cpos")
            print(str(plotter.camera_position))

        if print_cpos:
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        plotter.show()

    def get_dual_mesh_plotter(self, show_labels: bool = False, explode_factor: float = 0.0, exclude_boundaries: bool = False,
                              off_screen: bool = True, point_size: int = 15) -> pyvista.plotting.Plotter:
        mesh = self.dual_mesh
        if exclude_boundaries:
            boundary_nodes = [index for index in range(mesh.n_points) if self.get_dual_node(index).is_boundary]
            mesh, _ = mesh.remove_points(boundary_nodes)
        if explode_factor != 0.0:
            mesh = self.explode(mesh, explode_factor)
        plt = pyvista.plotting.Plotter(theme=self.pyvista_theme, off_screen=off_screen)
        if show_labels:
            plt.add_point_labels(mesh, "point_labels", point_size=point_size, font_size=20)
        plt.add_mesh(mesh, color="lightblue", smooth_shading=True)
        plt.add_points(mesh.points, scalars=mesh["colors"], point_size=point_size, show_scalar_bar=False,
                       clim=Color.color_limits())
        light = pyvista.Light(light_type='headlight')
        light.intensity = 0.8
        plt.add_light(light)
        return plt

    def show_debug_mesh(self, mesh: pyvista.PolyData, show_labels: bool = False, exclude_boundaries: bool = False,
                        point_size: int = 15, line_width: int = 1, print_cpos: bool = False,
                        initial_cpos = None) -> None:
        plotter = self.get_debug_mesh_plotter(mesh, show_labels, exclude_boundaries, off_screen=False,
                                              point_size=point_size, line_width=line_width)

        def my_cpos_callback(*args):
            # plotter.add_text(str(plotter.camera_position), name="cpos")
            print(str(plotter.camera_position))

        if print_cpos:
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        if initial_cpos:
            plotter.camera_position = initial_cpos
        plotter.show()

    def get_debug_mesh_plotter(self, mesh: pyvista.PolyData, show_labels: bool = False, exclude_boundaries: bool = False,
                               off_screen: bool = True, point_size: int = 120, line_width: int = 10, edge_color: str = None) -> pyvista.plotting.Plotter:
        if exclude_boundaries:
            boundary_indices = {index for index, is_boundary in enumerate(mesh["is_boundary"]) if is_boundary}
            new_indices = {old: new for old, new in zip(
                [i for i in range(len(mesh.points)) if i not in boundary_indices],
                range(len(mesh.points) - len(boundary_indices))
            )}
            points = [point for index, point in enumerate(mesh.points) if index not in boundary_indices]
            lines = [[new_indices[index] for index in line] for line in reconvert_faces(mesh.lines) if set(line).isdisjoint(boundary_indices)]
            labels = [label for index, label in enumerate(mesh["point_labels"]) if index not in boundary_indices]
            colors = [color for index, color in enumerate(mesh.point_data["colors"]) if index not in boundary_indices]
            edge_colors = [color for color, line in zip(mesh.cell_data["edge_colors"], reconvert_faces(mesh.lines))
                           if set(line).isdisjoint(boundary_indices)]
            edge_index = [index for index, line in zip(mesh.cell_data["edge_index"], reconvert_faces(mesh.lines))
                          if set(line).isdisjoint(boundary_indices)]
            mesh = pyvista.PolyData(points, lines=convert_faces(lines))
            mesh["point_labels"] = labels
            mesh.point_data["colors"] = colors
            mesh.cell_data["edge_colors"] = edge_colors
            mesh.cell_data["edge_index"] = edge_index
        plt = pyvista.plotting.Plotter(theme=self.pyvista_theme, off_screen=off_screen)
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
        return plt

    def show_primary_mesh(self, show_qubit_labels: bool = False, explode_factor: float = 0.0, print_cpos: bool = False,
                          highlighted_volumes: list[DualGraphNode] = None, highlighted_qubits: list[int] = None,
                          qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                          only_faces_with_color: Color | list[Color] = None, only_nodes_with_color: Color | list[Color] = None,
                          lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                          mandatory_face_qubits: set[int] = None, string_operator_qubits: set[int] = None, line_width: float = None,
                          initial_cpos = None, show_normal_qubits: bool = True, wireframe_plot: bool = False, transparent_faces: bool = False,
                          mandatory_cell_qubits: set[int] = None,
                          ) -> None:
        plotter = self.get_primary_mesh_plotter(
            show_qubit_labels, explode_factor, off_screen=False, highlighted_volumes=highlighted_volumes,
            highlighted_qubits=highlighted_qubits, qubit_coordinates=qubit_coordinates, point_size=15,
            only_faces_with_color=only_faces_with_color, only_nodes_with_color=only_nodes_with_color,
            lowest_title=lowest_title, highest_title=highest_title, mandatory_face_qubits=mandatory_face_qubits,
            string_operator_qubits=string_operator_qubits, show_normal_qubits=show_normal_qubits,
            line_width=line_width, wireframe_plot=wireframe_plot, transparent_faces=transparent_faces,
            mandatory_cell_qubits=mandatory_cell_qubits
        )

        def my_cpos_callback(*args):
            # plotter.add_text(str(plotter.camera_position), name="cpos")
            print(str(plotter.camera_position))

        if print_cpos:
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        if initial_cpos:
            plotter.camera_position = initial_cpos
        plotter.show()

    def get_primary_mesh_plotter(self, show_qubit_labels: bool = False, explode_factor: float = 0.0,
                                 off_screen: bool = True, point_size: int = 70, highlighted_volumes: list[DualGraphNode] = None,
                                 highlighted_qubits: list[int] = None, qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                                 only_faces_with_color: Color | list[Color] = None, only_nodes_with_color: Color | list[Color] = None,
                                 lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                                 mandatory_face_qubits: set[int] = None,  string_operator_qubits: set[int] = None, color_edges: bool = False,
                                 show_normal_qubits: bool = True, line_width: float = 3, wireframe_plot: bool = False,
                                 transparent_faces: bool = False, mandatory_cell_qubits: set[int] = None,
        ) -> pyvista.plotting.Plotter:
        """Return the plotter preloaded with the primary mesh.

        :param highlighted_volumes: Change color of given volumes to highlighted color. Take care to adjust the cmap of
            pyvista_theme to 'Color.highlighted_color_map' (otherwise there will be no visible effect).
        """
        highlighted_qubits = highlighted_qubits or []
        if highlighted_volumes or qubit_coordinates or only_faces_with_color or only_nodes_with_color or lowest_title or highest_title or mandatory_face_qubits or string_operator_qubits or color_edges or mandatory_cell_qubits:
            mesh = self._construct_primary_mesh(highlighted_volumes, qubit_coordinates, only_faces_with_color,
                                                only_nodes_with_color, lowest_title, highest_title, mandatory_face_qubits,
                                                string_operator_qubits, color_edges, mandatory_cell_qubits)
        else:
            mesh = self.primary_mesh
        if explode_factor != 0.0:
            mesh = self.explode(mesh, explode_factor)
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
        used_qubit_pos = set()
        if not explode_factor:
            used_qubit_pos = set(itertools.chain.from_iterable(reconvert_faces(mesh.faces)))
            if string_operator_qubits:
                used_qubit_pos.update([pos for pos, qubit in enumerate(mesh.point_data['qubits']) if qubit in string_operator_qubits])
        if show_normal_qubits:
            normal_qubits = set(mesh.point_data['qubits']) - set(highlighted_qubits)
        else:
            normal_qubits = set()
        for qubits, color in [(normal_qubits, "indigo"), (highlighted_qubits, "violet")]:
            positions = [pos for pos, qubit in enumerate(mesh.point_data['qubits']) if qubit in qubits and pos in used_qubit_pos]
            coordinates = np.asarray([coordinate for pos, coordinate in enumerate(mesh.points) if pos in positions])
            if len(coordinates) == 0:
                continue
            plt.add_points(coordinates, point_size=point_size, color=color)
            if show_qubit_labels:
                qubit_labels = [f"{qubit}" for pos, qubit in enumerate(mesh.point_data['qubits']) if pos in positions]
                plt.add_point_labels(coordinates, qubit_labels, show_points=False, font_size=20)
        if not color_edges:
            plt.add_mesh(mesh, show_scalar_bar=False, color="black", smooth_shading=True, line_width=line_width, style='wireframe')
        if not wireframe_plot:
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
        if self.dimension == 3 and not wireframe_plot and not transparent_faces:
            plt.remove_all_lights()
        return plt

    def show_debug_primary_meshes(self, mesh: pyvista.PolyData, show_qubit_labels: bool = False, explode_factor: float = 0.0,
                                  highlighted_volumes: list[DualGraphNode] = None, highlighted_qubits: list[int] = None,
                                  qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                                  only_faces_with_color: Color | list[Color] = None,
                                  only_nodes_with_color: Color | list[Color] = None,
                                  lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                                  mandatory_face_qubits: set[int] = None, string_operator_qubits: set[int] = None,
                                  wireframe_plot: bool = False, transparent_faces: bool = False,
                                  show_normal_qubits: bool = True, mesh_line_width: int = 1, primary_line_width: int = 1,
                                  highlighted_edges: list[GraphEdge] = None, show_normal_edges: bool = True,
                                  highlighted_line_width: int = 1, mandatory_cell_qubits: set[int] = None,
                                  initial_cpos=None, print_cpos: bool = False,
                                  ) -> None:
        plotter = self.get_debug_primary_meshes_plotter(
            mesh, show_qubit_labels, explode_factor, off_screen=False, highlighted_volumes=highlighted_volumes,
            highlighted_qubits=highlighted_qubits, qubit_coordinates=qubit_coordinates, qubit_point_size=15,
            only_faces_with_color=only_faces_with_color, only_nodes_with_color=only_nodes_with_color,
            lowest_title=lowest_title, highest_title=highest_title, mandatory_face_qubits=mandatory_face_qubits,
            string_operator_qubits=string_operator_qubits, show_normal_qubits=show_normal_qubits,
            wireframe_plot=wireframe_plot, transparent_faces=transparent_faces,
            highlighted_edges=highlighted_edges, show_normal_edges=show_normal_edges,
            node_point_size=20, mesh_line_width=mesh_line_width, primary_line_width=primary_line_width,
            highlighted_line_width=highlighted_line_width, mandatory_cell_qubits=mandatory_cell_qubits)

        def my_cpos_callback(*args):
            # plotter.add_text(str(plotter.camera_position), name="cpos")
            print(str(plotter.camera_position))

        if print_cpos:
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        if initial_cpos:
            plotter.camera_position = initial_cpos
        plotter.show()

    def get_debug_primary_meshes_plotter(self, mesh: pyvista.PolyData, show_qubit_labels: bool = False, explode_factor: float = 0.0,
                                         off_screen: bool = True, qubit_point_size: int = 70, highlighted_volumes: list[DualGraphNode] = None,
                                         highlighted_qubits: list[int] = None, qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None,
                                         only_faces_with_color: Color | list[Color] = None, only_nodes_with_color: Color | list[Color] = None,
                                         lowest_title: tuple[int, int, int] = None, highest_title: tuple[int, int, int] = None,
                                         mandatory_face_qubits: set[int] = None,  string_operator_qubits: set[int] = None, color_edges: bool = False,
                                         show_normal_qubits: bool = True, wireframe_plot: bool = False, transparent_faces: bool = False,
                                         highlighted_edges: list[GraphEdge] = None, mesh_line_width: int = 10, node_point_size: int = 120,
                                         show_normal_edges: bool = True, primary_line_width: int = None, highlighted_line_width: int = None,
                                         mesh_line_color: Optional[str] = None, mandatory_cell_qubits: set[int] = None) -> pyvista.plotting.Plotter:
        """Return the plotter preloaded with the debug and primary mesh.

        :param highlighted_edges: Edges of the debug graph to highlight.
        """
        highlighted_edges = highlighted_edges or []
        if explode_factor != 0.0:
            mesh = self.explode(mesh, explode_factor)
        plt = self.get_primary_mesh_plotter(show_qubit_labels, explode_factor, off_screen, qubit_point_size,
                                            highlighted_volumes, highlighted_qubits, qubit_coordinates,
                                            only_faces_with_color,only_nodes_with_color, lowest_title, highest_title,
                                            mandatory_face_qubits, string_operator_qubits, color_edges, show_normal_qubits,
                                            primary_line_width, wireframe_plot, transparent_faces, mandatory_cell_qubits)
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
        plt.add_points(mesh.points, scalars=mesh["colors"], point_size=node_point_size, show_scalar_bar=False,
                      clim=Color.color_limits())
        plt.enable_anti_aliasing('msaa', multi_samples=16)
        return plt


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

    def _construct_primary_mesh(self, highlighted_volumes: list[DualGraphNode] = None,
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
    def _dual_mesh_coordinates_from_primary_mesh_boundary_factor(distance: int) -> Optional[float]:
        return {
            4: 1.2,
            6: 0.96,
        }.get(distance)
