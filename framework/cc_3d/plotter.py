"""Plotting dual lattice & constructed primary lattice from graph definition."""
import dataclasses
import itertools
import pathlib
import re
from collections import defaultdict
from tempfile import NamedTemporaryFile

import numpy as np
import numpy.typing as npt
import pyvista
import pyvista.plotting
import pyvista.plotting.themes
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from scipy.spatial import Delaunay
import vtk

from framework.base import DualGraphNode, GraphNode
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


@dataclasses.dataclass
class Plotter3D:
    dual_graph: rx.PyGraph
    _dual_mesh: pyvista.PolyData = dataclasses.field(default=None, init=False)
    _primary_mesh: pyvista.PolyData = dataclasses.field(default=None, init=False)
    storage_dir: pathlib.Path = dataclasses.field(default=pathlib.Path(__file__).parent.parent.absolute())
    pyvista_theme: pyvista.plotting.themes.DocumentTheme = dataclasses.field(default=None, init=False)
    highes_id: int = dataclasses.field(default=0, init=False)
    _dualmesh_to_dualgraph: dict[int, int] = dataclasses.field(default=None, init=False)

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
        theme.cmap = Color.color_map()
        theme.show_vertices = True
        theme.show_edges = True
        theme.lighting = 'none'
        theme.render_points_as_spheres = True
        theme.render_lines_as_tubes = True
        return theme

    @property
    def next_id(self) -> int:
        """Generate the next unique id to label mesh objects."""
        self.highes_id += 1
        return self.highes_id

    def get_dual_node(self, mesh_index: int) -> DualGraphNode:
        return self.dual_graph[self._dualmesh_to_dualgraph[mesh_index]]

    @staticmethod
    def get_3d_coordinates(graph: rx.PyGraph) -> dict[int, npt.NDArray[np.float64]]:
        """Calculate 3D coordinates of nodes by layouting the rustworkx graph.

        Take special care to place boundary nodes at a meaningful position.

        :returns: Mapping of rx node indices to [x, y, z] coordinates.
        """
        # remove boundary nodes for bulk positioning
        graph_without_boundaries = graph.copy()
        boundary_node_indices = [node.index for node in graph_without_boundaries.nodes() if node.is_boundary]
        graph_without_boundaries.remove_nodes_from(boundary_node_indices)

        with NamedTemporaryFile("w+t", suffix=".wrl") as f:
            graphviz_draw(graph_without_boundaries, lambda node: {"shape": "point"}, filename=f.name, method="neato",
                          image_type="vrml", graph_attr={"dimen": "3"})
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

    def _construct_dual_mesh(self) -> pyvista.PolyData:
        # calculate positions of points
        node2coordinates = self.get_3d_coordinates(self.dual_graph)
        points = np.asarray([node2coordinates[index] for index in self.dual_graph.node_indices()])

        # generate pyvista edges from rustworkx edges
        rustworkx2pyvista = {rustworkx_index: pyvista_index for pyvista_index, rustworkx_index in enumerate(self.dual_graph.node_indices())}
        self._dualmesh_to_dualgraph = {value: key for key, value in rustworkx2pyvista.items()}
        # TODO ensure all faces of dual graph are triangles?
        simplexes = compute_simplexes(self.dual_graph, dimension=3, exclude_boundary_simplexes=True)
        # each simplex (tetrahedron) has four faces (triangles)
        faces = [[rustworkx2pyvista[index] for index in combination] for simplex in simplexes for combination in itertools.combinations(simplex, 3)]

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
        tetrahedron_ids = itertools.chain.from_iterable([[self.next_id]*4 for _ in simplexes])
        ret.cell_data["face_ids"] = list(tetrahedron_ids)

        # add the qubit to each face of its tetrahedron
        labels = []
        for simplex in simplexes:
            qubits = set(self.dual_graph.nodes()[simplex[0]].qubits)
            for index in simplex[1:]:
                qubits &= set(self.dual_graph.nodes()[index].qubits)
            if len(qubits) != 1:
                raise RuntimeError
            labels.extend([qubits.pop()] * len(simplex))
        ret.cell_data["qubits"] = labels

        # add color
        colors = [node.color for node in self.dual_graph.nodes()]
        ret["colors"] = colors

        return ret

    def construct_debug_mesh(self, graph: rx.PyGraph, coordinates: dict[int, npt.NDArray[np.float64]] = None,
                             use_edges_colors: bool = False, highlighted_nodes: list[GraphNode] = None) -> pyvista.PolyData:
        """Create a 3D mesh of the given rustworkx Graph.

        Nodes must be GraphNode and edges GraphEdge objects.

        :param coordinates: Mapping of node indices of graph to 3D coordinates. Use them instead of calculating them.
        :param use_edges_colors: If true, use the color of the GraphEdge object instead of default colors.
        :param highlighted_nodes: Change color of given nodes to highlighted color. Take care to adjust the cmap of
            pyvista_theme to 'Color.highlighted_color_map' (otherwise there will be no visible effect).
        """
        highlighted_nodes = highlighted_nodes or []
        # calculate positions of points (or use given coordinates)
        node2coordinates = coordinates or self.get_3d_coordinates(graph)
        points = np.asarray([node2coordinates[index] for index in graph.node_indices()])

        # generate pyvista edges from rustworkx edges
        rustworkx2pyvista = {rustworkx_index: pyvista_index for pyvista_index, rustworkx_index in enumerate(graph.node_indices())}
        lines = [[rustworkx2pyvista[node1], rustworkx2pyvista[node2]] for node1, node2, _ in graph.edge_index_map().values()]

        ret = pyvista.PolyData(points, lines=convert_faces(lines))

        # remember which nodes are boundary nodes
        boundaries = [node.is_boundary for node in graph.nodes()]
        ret["is_boundary"] = boundaries

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
            if use_edges_colors:
                # use grey as fallback
                edge_colors.append(edge.color if edge.color is not None else Color.by)
                continue
            if edge.is_edge_between_boundaries:
                edge_colors.append(Color.red)
            elif edge.node1.is_boundary or edge.node2.is_boundary:
                edge_colors.append(Color.green)
            else:
                # dark blue
                edge_colors.append(Color.bg)
        ret.cell_data["edge_colors"] = edge_colors

        # add colors to points
        colors = [node.color.highlight if node in highlighted_nodes else node.color for node in graph.nodes()]
        ret.point_data["colors"] = colors

        return ret

    def _construct_primary_mesh(self, highlighted_volumes: list[DualGraphNode] = None,
                                qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None):
        """Construct primary mesh from dual_mesh.

        :param qubit_coordinates: Use them instead of calculating the coordinates from the dual_mesh.
        """
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
            # exactly two nodes are boundary nodes
            if sum(node.is_boundary for node in dg_nodes) == 2:
                # move the point a bit more outward (away from the center)
                center += 0.1 * (center - dual_mesh_center)
            # exactly three nodes are boundary nodes
            elif sum(node.is_boundary for node in dg_nodes) == 3:
                center += 0.3 * (center - dual_mesh_center)
            # use given coordinates if provided
            if qubit_coordinates:
                points.append(qubit_coordinates[qubit])
                qubit_to_point[qubit] = qubit_coordinates[qubit]
            else:
                points.append(center)
                qubit_to_point[qubit] = center
            qubit_to_pointposition[qubit] = pointposition
            qubits.append(qubit)

        # vertices -> volumes
        volumes = []
        volume_ids = []
        volume_colors = []
        for node in self.dual_graph.nodes():
            # do not add a volume for boundaries
            if node.is_boundary:
                continue
            # each dual graph edge corresponds to a primary graph face
            all_face_qubits = [edge.qubits for _, _, edge in self.dual_graph.out_edges(node.index)]
            faces = []
            for face_qubits in all_face_qubits:
                face_points = [qubit_to_point[qubit] for qubit in face_qubits]
                # project the points to a 2D plane with 2D coordinates, then calculate their triangulation
                triangulation = Delaunay(project_to_plane(face_points), qhull_options="QJ")
                # extract faces of the triangulation, take care to use the qubits
                tmp_point_map = {k: v for k, v in zip(range(triangulation.npoints), face_qubits)}
                simplexes = [[tmp_point_map[point] for point in face] for face in triangulation.simplices]
                face = [qubit_to_pointposition[qubit] for qubit in triangles_to_face(simplexes)]
                faces.append(face)
            # add volume faces
            volumes.extend(faces)
            # add volume ids
            volume_ids.extend([self.next_id] * len(faces))
            # add volume colors
            volume_colors.extend([node.color.highlight if node in highlighted_volumes else node.color] * len(faces))
        ret = pyvista.PolyData(points, faces=convert_faces(volumes))
        ret.point_data['qubits'] = qubits
        ret.cell_data['face_ids'] = volume_ids
        ret.cell_data['colors'] = volume_colors
        return ret

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
        plt = pyvista.plotting.Plotter(theme=self.pyvista_theme, lighting='none', off_screen=off_screen)
        plt.disable_shadows()
        plt.disable_ssao()
        if show_labels:
            plt.add_point_labels(mesh, "point_labels", point_size=point_size, font_size=20)
        plt.add_mesh(mesh, color="lightblue")
        plt.add_points(mesh.points, scalars=mesh["colors"], render_points_as_spheres=True, point_size=point_size,
                       show_scalar_bar=False, clim=Color.color_limits())
        return plt

    def show_debug_mesh(self, mesh: pyvista.PolyData, show_labels: bool = False, exclude_boundaries: bool = False,
                        point_size: int = 15, line_width: int = 1, print_cpos: bool = False) -> None:
        plotter = self.get_debug_mesh_plotter(mesh, show_labels, exclude_boundaries, off_screen=False,
                                              point_size=point_size, line_width=line_width)

        def my_cpos_callback(*args):
            # plotter.add_text(str(plotter.camera_position), name="cpos")
            print(str(plotter.camera_position))

        if print_cpos:
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        plotter.show()

    def get_debug_mesh_plotter(self, mesh: pyvista.PolyData, show_labels: bool = False, exclude_boundaries: bool = False,
                               off_screen: bool = True, point_size: int = 15, line_width: int = 1) -> pyvista.plotting.Plotter:
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
            mesh = pyvista.PolyData(points, lines=convert_faces(lines))
            mesh["point_labels"] = labels
            mesh.point_data["colors"] = colors
            mesh.cell_data["edge_colors"] = edge_colors
        plt = pyvista.plotting.Plotter(theme=self.pyvista_theme, lighting='none', off_screen=off_screen)
        plt.disable_shadows()
        plt.disable_ssao()
        if show_labels:
            plt.add_point_labels(mesh, "point_labels", point_size=point_size, font_size=20)
        plt.add_mesh(mesh, scalars="edge_colors", show_scalar_bar=False, cmap=Color.color_map(),
                     clim=Color.color_limits(), line_width=line_width)
        plt.add_points(mesh.points, scalars=mesh["colors"], render_points_as_spheres=True, point_size=point_size,
                       show_scalar_bar=False, clim=Color.color_limits())
        return plt

    def show_primary_mesh(self, show_qubit_labels: bool = False, explode_factor: float = 0.0, print_cpos: bool = False,
                          highlighted_volumes: list[DualGraphNode] = None, highlighted_qubits: list[int] = None,
                          qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None) -> None:
        plotter = self.get_primary_mesh_plotter(
            show_qubit_labels, explode_factor, off_screen=False, highlighted_volumes=highlighted_volumes,
            highlighted_qubits=highlighted_qubits, qubit_coordinates=qubit_coordinates)

        def my_cpos_callback(*args):
            # plotter.add_text(str(plotter.camera_position), name="cpos")
            print(str(plotter.camera_position))

        if print_cpos:
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        plotter.show()

    def get_primary_mesh_plotter(self, show_qubit_labels: bool = False, explode_factor: float = 0.0,
                                 off_screen: bool = True, point_size: int = 15, highlighted_volumes: list[DualGraphNode] = None,
                                 highlighted_qubits: list[int] = None, qubit_coordinates: dict[int, npt.NDArray[np.float64]] = None
                                 ) -> pyvista.plotting.Plotter:
        """Return the plotter preloaded with the primary mesh.

        :param highlighted_volumes: Change color of given volumes to highlighted color. Take care to adjust the cmap of
            pyvista_theme to 'Color.highlighted_color_map' (otherwise there will be no visible effect).
        """
        if highlighted_volumes or qubit_coordinates:
            mesh = self._construct_primary_mesh(highlighted_volumes, qubit_coordinates)
        else:
            mesh = self.primary_mesh
        if explode_factor != 0.0:
            mesh = self.explode(mesh, explode_factor)
        plt = pyvista.plotting.Plotter(theme=self.pyvista_theme, lighting='none', off_screen=off_screen)
        plt.disable_shadows()
        plt.disable_ssao()
        if highlighted_qubits:
            positions = [pos for pos, qubit in enumerate(mesh.point_data['qubits']) if qubit in highlighted_qubits]
            coordinates = np.asarray([coordinate for pos, coordinate in enumerate(mesh.points) if pos in positions])
            plt.add_points(coordinates, point_size=point_size, color="magenta")
        if show_qubit_labels:
            plt.add_point_labels(mesh, "qubits", point_size=point_size, font_size=20)
        plt.add_mesh(mesh, scalars="colors", show_scalar_bar=False, clim=Color.color_limits())
        return plt
