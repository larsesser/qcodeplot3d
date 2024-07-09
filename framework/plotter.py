"""Plotting dual lattice & constructed primary lattice from graph definition."""
import dataclasses
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Union

import pyvista
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import itertools
import re
import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
from scipy.linalg import det
import scipy.spatial


# see https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydata.n_faces#pyvista.PolyData.n_faces
pyvista.PolyData.use_strict_n_faces(True)


def convert_faces(faces: list[list[int]]) -> list[int]:
    """Pad a list of faces so that pyvista can process it."""
    return list(itertools.chain.from_iterable([(len(face), *face) for face in faces]))


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


def to_tuple(iterable: Union[Iterable[Any], Any]) -> Union[Any, tuple[Any, ...]]:
    if isinstance(iterable, Iterable):
        return tuple(to_tuple(e) for e in iterable)
    return iterable


def triangles2faces(triangles_: list[list[int]], positions2points: dict[int, np.ndarray]) -> list[list[int]]:
    """Converts a triangulation of an object to the proper planes of this object.

    :param triangles: list of all simplicies (triangles) by point position.
    :param positions2points: maps each point position to an actual 3D coordinate.
    """
    if any(len(triangle) != 3 for triangle in triangles_):
        raise ValueError
    triangles = to_tuple(triangles_)

    neighboured = defaultdict(list)
    # calculate which triangles are neighboured and at the same face
    for triangle1, triangle2 in itertools.combinations(triangles, 2):
        # first, determine if the two triangles share an edge. This is rather
        # easy, since all points of a triangle are connected, so we only need
        # to check if both triangles share at least two points.
        if len(shared := set(triangle1) & set(triangle2)) < 2:
            continue
        # second, determine if the two triangles lay in the same plane.
        # to check this, we
        # - shift one point to (0, 0, 0) by subtracting it from all other points
        # - take the remaining 3 points and check if they do _not_ span the whole
        #   space by checking if their determinante is 0
        a = positions2points[(set(triangle1) - shared).pop()]
        b = positions2points[(set(triangle2) - shared).pop()] - a
        c = positions2points[shared.pop()] - a
        d = positions2points[shared.pop()] - a
        if det(np.asarray([b, c, d])) != 0:
            continue
        neighboured[triangle1].append(triangle2)
        neighboured[triangle2].append(triangle1)

    consumed: set[tuple[int, int, int]] = set()

    def construct_face(triangle: tuple[int, int, int]) -> list[int]:
        """Recursive function to construct one face from all neighbouring triangles."""
        ret = [*triangle]
        # print(f"Construct with {triangle}")
        consumed.add(triangle)
        neighbours = neighboured[triangle]

        for neighbour in neighbours:
            # take care to take each triangle only once into account
            if neighbour in consumed:
                continue

            # construct the face of all neighbours of this neighbour
            subface = construct_face(neighbour)
            # print(f"\nBack in {triangle} from {neighbour}")
            # print(f"Subface: {subface}, ret: {ret}")
            # the subface and ret share exactly one edge
            shared = set(ret) & set(subface)
            if len(shared) != 2:
                raise RuntimeError
            shared_a = shared.pop()
            shared_b = shared.pop()
            # sort both points by their appearance in ret
            # if ret.index(shared_a) > ret.index(shared_b):
            #     shared_a, shared_b = shared_b, shared_a
            # print("Shared: ", shared_a, shared_b)

            # sort ret in such a way that it starts with shared_a
            new_ret = []
            iterator = itertools.cycle(ret)
            while len(new_ret) != len(ret):
                element = next(iterator)
                if element == shared_a or new_ret:
                    new_ret.append(element)
            # print(f"New ret: {new_ret}")

            # now there are two possible layouts of new_ret:
            # 1. [shared_a, ..., shared_b]
            # 2. [shared_a, shared_b, ...]

            # sort subface in such a way that it starts with shared_a
            new_subface = []
            iterator = itertools.cycle(subface)
            while len(new_subface) != len(subface):
                element = next(iterator)
                if element == shared_a or new_subface:
                    new_subface.append(element)
            # print(f"New subface: {new_subface}")

            # now there are two possible layouts of new_subface:
            # 1. [shared_a, ..., shared_b]
            # 2. [shared_a, shared_b, ...]

            # finally, include new_subface into new_ret, depending on the cases:
            # 1. and 1.: reverse remainder and append it to new_ret
            if new_ret[-1] == shared_b and new_subface[-1] == shared_b:
                remainder = new_subface[1:-1][::-1]
                new_ret.extend(remainder)
            # 1. and 2.: append remainder to new_ret
            elif new_ret[-1] == shared_b and new_subface[-1] != shared_b:
                remainder = new_subface[2:]
                new_ret.extend(remainder)
            # 2. and 1.: insert remainder between shared_a and shared_b
            elif new_ret[-1] != shared_b and new_subface[-1] == shared_b:
                remainder = new_subface[1:-1]
                new_ret = [new_ret[0]] + remainder + new_ret[1:]
            # 2. and 2.: reverse remainder and insert it between shared_a and shared_b
            elif new_ret[-1] != shared_b and new_subface[-1] != shared_b:
                remainder = new_subface[2:][::-1]
                new_ret = [new_ret[0]] + remainder + new_ret[1:]
            else:
                raise RuntimeError
            # print(f"Remainder: {remainder}")
            # print(f"New Ret: {new_ret}")
            ret = new_ret
        return ret

    ret2 = []
    for triangle in triangles:
        if triangle in consumed:
            continue
        ret2.append(construct_face(triangle))
    return ret2


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
    primary_mesh: pyvista.PolyData = dataclasses.field(init=False)
    name: str
    storage_dir: pathlib.Path = dataclasses.field(default=pathlib.Path(__file__).parent.parent.absolute())
    highes_tetrahedron_id: int = dataclasses.field(default=0, init=False)

    def __post_init__(self):
        self.dual_mesh = self._dual_mesh_from_graph()
        self.primary_mesh = self._construct_primary_mesh()

    @property
    def next_tetrahedron_id(self) -> int:
        """Generate the next unique id for a tetrahedron object."""
        self.highes_tetrahedron_id += 1
        return self.highes_tetrahedron_id

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

        # add tetrahedron ids
        tetrahedron_ids = itertools.chain.from_iterable([[self.next_tetrahedron_id]*4 for _ in simplexes])
        ret.cell_data["tetrahedron_ids"] = list(tetrahedron_ids)

        return ret

    def _construct_primary_mesh(self):
        # group dual lattice cells by tetrahedron
        volumes2facepositions = defaultdict(list)
        for faceposition, anid in enumerate(self.dual_mesh.cell_data['tetrahedron_ids']):
            volumes2facepositions[anid].append(faceposition)

        # group dual lattice faces by points in primal lattice
        # note that this uses faces and not volumes as key, to simplify the construction
        faces2points = defaultdict(list)

        # volumes -> vertices
        points = []
        all_faces = reconvert_faces(self.dual_mesh.faces)
        for facepositions in volumes2facepositions.values():
            tetrahedron = self.dual_mesh.extract_cells(facepositions)
            # find center of mass of the tetrahedron
            center = np.asarray([0.0, 0.0, 0.0])
            for point in tetrahedron.points:
                center += point
            center = center / len(tetrahedron.points)
            faces = [all_faces[position] for position in facepositions]
            for face in faces:
                faces2points[tuple(sorted(face))].append(len(points))
            points.append(center)

        # vertices -> volumes
        volumes = []
        volume_ids = []
        for point in range(self.dual_mesh.n_points):
            volume_positions = set()
            for face in all_faces:
                # if len(face) != 3:
                #     raise RuntimeError
                if point in face:
                    volume_positions.update(set(faces2points[tuple(sorted(face))]))
            volume_positions = sorted(volume_positions)
            # only look at relevant volumes, the other ones are boundary related
            # if len(volume_positions) not in [8, 32]:
            #     continue
            volume_points = [points[position] for position in volume_positions]
            volume_positions2points = {position: points[position] for position in volume_positions}
            # create the convex hull of all points of this volume
            try:
                hull = ConvexHull(volume_points)
            except scipy.spatial.QhullError as e:
                print(e)
                continue
            # extract faces of the hull, take care to use the original points
            tmp_point_map = {k: v for k, v in zip(range(hull.npoints), volume_positions)}
            simplices = [[tmp_point_map[point] for point in face] for face in hull.simplices]
            faces = triangles2faces(simplices, volume_positions2points)
            # again to filter out some boundary stuff
            if len(faces) not in {6, 18}:
                continue
            volumes.extend(faces)

            # add volume ids
            # new_id = max(THETHRAEDA_IDS or [0]) + 1
            # volume_ids.extend([new_id] * len(faces))
            # THETHRAEDA_IDS.append(new_id)

        print(volumes)
        ret = pyvista.PolyData(points, faces=convert_faces(volumes))
        # ret.cell_data['tetrahedron_ids'] = volume_ids
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
