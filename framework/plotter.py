"""Plotting dual lattice & constructed primary lattice from graph definition."""
import dataclasses
import pathlib

import pyvista
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import itertools
import re
import numpy as np
import numpy.typing as npt


# see https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydata.n_faces#pyvista.PolyData.n_faces
pyvista.PolyData.use_strict_n_faces(True)


def convert_faces(faces: list[list[int]]) -> list[int]:
    """Pad a list of faces so that pyvista can process it."""
    return list(itertools.chain.from_iterable([(len(face), *face) for face in faces]))


def compute_simplexes(graph: rx.PyGraph, dimension: int) -> set[tuple[int, ...]]:
    """Find all simplexes of the given dimension in the graph."""
    if dimension not in {2, 3}:
        raise NotImplementedError
    triangles = set()
    for node1 in graph.nodes():
        node1_neighbors = graph.neighbors(node1.index)
        for node2_index in node1_neighbors:
            for node3_index in graph.neighbors(node2_index):
                if node3_index not in node1_neighbors:
                    continue
                triangles.add(tuple(sorted([node1.index, node2_index, node3_index])))
    if dimension == 2:
        return triangles
    tetrahedrons = set()
    for triangle in triangles:
        common_neighbours = set(graph.neighbors(triangle[0])) & set(graph.neighbors(triangle[1])) & set(graph.neighbors(triangle[2]))
        for neighbour in common_neighbours:
            tetrahedrons.add(tuple(sorted([*triangle, neighbour])))
    return tetrahedrons


@dataclasses.dataclass
class Plotter3D:
    dual_graph: rx.PyGraph
    dual_mesh: pyvista.PolyData = dataclasses.field(init=False)
    name: str
    storage_dir: pathlib.Path = dataclasses.field(default=pathlib.Path(__file__).parent.parent.absolute())

    def __post_init__(self):
        self.dual_mesh = self._dual_mesh_from_graph()

    def _get_3d_coordinates(self) -> dict[int, npt.NDArray[np.float64]]:
        """Calculate 3D coordinates of nodes by layouting the rustworkx graph.

        Take special care to place boundary nodes at a meaningful position.

        :returns: Mapping of rx node indices to [x, y, z] coordinates.
        """
        filename = self.name + ".wrl"
        path = self.storage_dir / filename

        # remove boundary nodes for bulk positioning
        graph = self.dual_graph.copy()
        boundary_node_indices = [node.index for node in graph.nodes() if node.is_boundary]
        graph.remove_nodes_from(boundary_node_indices)

        # TODO do we want to cache the result? I.e., change this to "if not path.is_file()"
        graphviz_draw(graph, lambda node: {"shape": "point"}, filename=str(path), method="neato", image_type="vrml",
                      graph_attr={"dimen": "3"})
        with open(path, "rt") as f:
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
            adjacent_nodes = [index for index in self.dual_graph.neighbors(boundary_index)
                              if not self.dual_graph[index].is_boundary]
            face_center = np.asarray([0.0, 0.0, 0.0])
            for index, position in ret.items():
                if index not in adjacent_nodes:
                    continue
                face_center += position
            face_center /= len(adjacent_nodes)

            # extrapolate the position of the boundary node from the line through center and face_center
            pos = face_center + (face_center - center)
            ret[boundary_index] = pos
        return ret

    def _dual_mesh_from_graph(self) -> pyvista.PolyData:
        # calculate positions of points
        node2coordinates = self._get_3d_coordinates()
        points = np.asarray([node2coordinates[index] for index in self.dual_graph.node_indices()])

        # generate pyvista edges from rustworkx edges
        rustworkx2pyvista = {rustworkx_index: pyvista_index for pyvista_index, rustworkx_index in enumerate(self.dual_graph.node_indices())}
        simplexes = compute_simplexes(self.dual_graph, dimension=3)
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

        return ret

    def show_dual_mesh(self, show_labels: bool = False) -> None:
        plt = pyvista.Plotter(lighting='none')
        plt.disable_shadows()
        plt.disable_ssao()
        plt.show_axes()
        if show_labels:
            plt.add_point_labels(self.dual_mesh, "point_labels", point_size=30, font_size=20)
        plt.add_mesh(self.dual_mesh)
        plt.show()
