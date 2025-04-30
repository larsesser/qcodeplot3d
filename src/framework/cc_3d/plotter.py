"""Plotting dual lattice & constructed primary lattice from graph definition."""
import abc
import collections
import dataclasses
import itertools
import re
from collections import defaultdict
from typing import ClassVar, Optional, Union

import numpy as np
import numpy.typing as npt
import pyvista
from scipy.spatial import Delaunay

from framework.common.graph import DualGraphNode
from framework.common.plotter import (
    Plotter,
    convert_faces,
    cross_point_2_lines,
    cross_point_3_planes,
    distance_between_points,
    distance_to_plane,
    project_to_2d_plane,
    project_to_3d_plane,
    project_to_given_plane,
    project_to_line,
    reconvert_faces,
    triangles_to_face,
)
from framework.common.stabilizers import Color


@dataclasses.dataclass
class Plotter3D(Plotter, abc.ABC):
    dimension: ClassVar[int] = 3

    def layout_primary_nodes(self, given_qubit_coordinates: dict[int, npt.NDArray[np.float64]]) -> (list[npt.NDArray[np.float64]], dict[int, int]):
        # volumes -> vertices

        points: list[npt.NDArray[np.float64]] = []
        # map each qubit to the position of its points coordinates
        qubit_to_pointpos: dict[int, int] = {}

        # group dual lattice cells (of the tetrahedron) by qubit
        qubit_to_facepositions: dict[int, list[int]] = defaultdict(list)
        for position, qubit in enumerate(self.dual_mesh.cell_data['qubits']):
            qubit_to_facepositions[qubit].append(position)

        for pointposition, (qubit, facepositions) in enumerate(qubit_to_facepositions.items()):
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

        points = self.postprocess_primary_node_layout(points, qubit_to_pointpos)
        return points, qubit_to_pointpos

    @abc.abstractmethod
    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int]
    ) -> list[npt.NDArray[np.float64]]:
        ...

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
                    if edge not in lines and (edge[1], edge[0]) not in lines and (0.9*distance <= min_distance or self.dimension == 3):
                        lines.append(edge)
                        line_ids.append(self.next_id)
                        line_colors.append(volume_colors_by_pos[pos1].highlight)
        ret = pyvista.PolyData(points, faces=convert_faces(volumes), lines=convert_faces(lines) if len(lines) else None)
        ret.point_data['qubits'] = qubits
        ret.cell_data['face_ids'] = [*line_ids, *volume_ids]
        ret.cell_data['colors'] = [*line_colors, *volume_colors]
        ret.cell_data['pyvista_indices'] = [*([-1]*len(lines)), *volume_indices]
        return ret


@dataclasses.dataclass
class TetrahedronPlotter(Plotter3D):
    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int]
    ) -> list[npt.NDArray[np.float64]]:
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}
        corner_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 3}
        border_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 2}
        border_to_qubit: dict[tuple[int, int]: list[int]] = collections.defaultdict(list)
        for qubit, nodes in border_qubits.items():
            border_to_qubit[tuple(sorted([nodes[0].index, nodes[1].index]))].append(qubit)
        boundary_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 1}
        boundary_to_qubit: dict[int, list[int]] = collections.defaultdict(list)
        for qubit, nodes in boundary_qubits.items():
            boundary_to_qubit[nodes[0].index].append(qubit)
        bulk_qubits = {qubit for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 0}


        if self.distance == 3:
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

            # project qubits to the border
            line_qubits = list(set(boundary1.qubits) & set(boundary2.qubits) & set(corner_qubits))
            if len(line_qubits) != 2:
                raise ValueError
            line = [qubit_to_point[line_qubits[0]], qubit_to_point[line_qubits[1]]]
            # place qubit for d=3 tetrahedron in center of border
            if self.distance == 3:
                qubit_to_point[qubits[0]] = points[qubit_to_pointpos[qubits[0]]] = (line[0] + line[1]) / 2
                continue
            to_plane = {qubit: coordinate for qubit, coordinate in zip(qubits, project_to_given_plane([line[0], line[1], dual_mesh_center], [qubit_to_point[qubit] for qubit in qubits]))}
            for qubit, coordinate in to_plane.items():
                coordinate = cross_point_2_lines(line, [dual_mesh_center, coordinate])
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

            # move first and last qubit of each border more towards the corner
            tmp = {qubit: distance_between_points(line[0], project_to_line(line, qubit_to_point[qubit])) for qubit in qubits}
            qubit_order = sorted(tmp, key=lambda x: tmp[x])
            qubit_to_point[qubit_order[0]] = points[qubit_to_pointpos[qubit_order[0]]] = (2*qubit_to_point[qubit_order[0]] + line[0]) / 3
            qubit_to_point[qubit_order[-1]] = points[qubit_to_pointpos[qubit_order[-1]]] = (2*qubit_to_point[qubit_order[-1]] + line[1]) / 3
            # TODO scale remaining qubits better
            # assign equal-spaced coordinates to each qubit
            # qubit_distance = (line[1] - line[0]) / (len(qubits) + 3)
            # for i, qubit in enumerate(qubit_order[1:-1], start=3):
            #     qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = line[0] + i*qubit_distance

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
            3: 1.5,
            5: 1.5,
        }.get(distance)

    def preprocess_dual_node_layout(self, primary_mesh: pyvista.PolyData):
        lines = reconvert_faces(primary_mesh.lines)
        faces = reconvert_faces(primary_mesh.faces)
        dg_index_to_points: dict[int, dict[int, npt.NDArray[np.float64]]] = defaultdict(dict)
        for dg_index, face in zip(primary_mesh.cell_data['pyvista_indices'][len(lines):], faces):
            for pos in face:
                dg_index_to_points[dg_index][primary_mesh.point_data['qubits'][pos]] = primary_mesh.points[pos]

        corner_qubits = {(set(node1.qubits) & set(node2.qubits) & set(node3.qubits)).pop()
                         for node1, node2, node3 in itertools.combinations(self.boundary_nodes, 3)}

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
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int]
    ) -> list[npt.NDArray[np.float64]]:
        """Move qubits at the outside more outward, to form an even plane at each boundary."""
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}
        corner_qubits = {qubit for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 3}
        border_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 2}

        # calculate the reference planes
        boundary_to_reference_plane: dict[int, list[np.ndarray]] = {}
        for node in self.boundary_nodes:
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
            4: 0.8,
            6: 1.0,
        }.get(distance)

    def preprocess_dual_node_layout(self, primary_mesh: pyvista.PolyData):
        lines = reconvert_faces(primary_mesh.lines)
        faces = reconvert_faces(primary_mesh.faces)
        dg_index_to_points: dict[int, dict[int, npt.NDArray[np.float64]]] = defaultdict(dict)
        for dg_index, face in zip(primary_mesh.cell_data['pyvista_indices'][len(lines):], faces):
            for pos in face:
                dg_index_to_points[dg_index][primary_mesh.point_data['qubits'][pos]] = primary_mesh.points[pos]

        corner_qubits = set()
        for node1, node2, node3 in itertools.combinations(self.boundary_nodes, 3):
            corner_qubits.update(set(node1.qubits) & set(node2.qubits) & set(node3.qubits))
        border_qubits = set()
        for node1, node2 in itertools.combinations(self.boundary_nodes, 2):
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
