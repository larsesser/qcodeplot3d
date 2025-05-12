"""Plotting dual lattice & constructed primary lattice from graph definition."""
import abc
import collections
import dataclasses
import itertools
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
    distance_between_points,
    project_to_given_plane,
    project_to_line,
    reconvert_faces,
    triangles_to_face,
)
from framework.common.stabilizers import Color


@dataclasses.dataclass
class Plotter2D(Plotter, abc.ABC):
    dimension: ClassVar[int] = 2

    def layout_primary_nodes(
            self, given_qubit_coordinates: dict[int, npt.NDArray[np.float64]],
    ) -> tuple[list[npt.NDArray[np.float64]], dict[int, int]]:
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
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int],
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
        """
        highlighted_faces = highlighted_volumes or []

        points, qubit_to_pointpos = self.layout_primary_nodes(qubit_coordinates)
        qubits = list(qubit_to_pointpos.keys())
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}

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
                faces_by_pos[pos] = {qubit_to_pointpos[qubit] for qubit in node.qubits}
                face_colors_by_pos[pos] = node.color
                continue
            # 2D coordinates are enough, z coordinate is always 0
            face_points = [(qubit_to_point[qubit][0], qubit_to_point[qubit][1]) for qubit in node.qubits]
            triangulation = Delaunay(face_points, qhull_options="QJ")
            # extract faces of the triangulation, take care to use the qubits
            tmp_point_map = {k: v for k, v in zip(range(triangulation.npoints), node.qubits)}
            simplexes = [[tmp_point_map[point] for point in face] for face in triangulation.simplices]
            face = [qubit_to_pointpos[qubit] for qubit in triangles_to_face(simplexes)]
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
        ret = pyvista.PolyData(points, faces=convert_faces(faces), lines=convert_faces(lines) if lines else None)
        ret.point_data['qubits'] = qubits
        ret.cell_data['face_ids'] = [*line_ids, *face_ids]
        ret.cell_data['colors'] = [*line_colors, *face_colors]
        ret.cell_data['pyvista_indices'] = [*([-1]*len(lines)), *face_indices]
        return ret


@dataclasses.dataclass
class TriangularPlotter(Plotter2D):
    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int],
    ) -> list[npt.NDArray[np.float64]]:
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}
        corner_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 2}
        border_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 1}
        border_to_qubit: dict[int: list[int]] = collections.defaultdict(list)
        for qubit, nodes in border_qubits.items():
            border_to_qubit[nodes[0].index].append(qubit)
        bulk_qubits = {qubit for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 0}

        # move corner qubits more inward
        dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
        for qubit in corner_qubits:
            dual_mesh_center += qubit_to_point[qubit]
        dual_mesh_center /= len(corner_qubits)
        for qubit in corner_qubits:
            coordinate = qubit_to_point[qubit] - 0.15 * (qubit_to_point[qubit] - dual_mesh_center)
            qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

        # move border qubits to the line spanned by their respective corner qubits, equally spaced
        for boundary_index, qubits in border_to_qubit.items():
            boundary = self.dual_graph[boundary_index]

            # project qubits to the border
            line_qubits = list(set(boundary.qubits) & set(corner_qubits))
            if len(line_qubits) != 2:
                raise ValueError
            line = [qubit_to_point[line_qubits[0]], qubit_to_point[line_qubits[1]]]
            # place qubit for d=3 tetrahedron in center of border
            if self.distance == 3:
                qubit_to_point[qubits[0]] = points[qubit_to_pointpos[qubits[0]]] = (line[0] + line[1]) / 2
                continue
            to_plane = {
                qubit: coordinate
                for qubit, coordinate in zip(qubits, project_to_given_plane(
                    [line[0], line[1], dual_mesh_center], [qubit_to_point[qubit] for qubit in qubits]))
            }
            for qubit, coordinate in to_plane.items():
                new_coordinate = cross_point_2_lines(line, [dual_mesh_center, coordinate])
                qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = new_coordinate

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

        # move bulk qubits more to center
        dual_mesh_center = np.asarray([0.0, 0.0, 0.0])
        for qubit in corner_qubits:
            dual_mesh_center += qubit_to_point[qubit]
        dual_mesh_center /= len(corner_qubits)
        for qubit in bulk_qubits:
            coordinate = qubit_to_point[qubit] - 0.5 * (qubit_to_point[qubit] - dual_mesh_center)
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

        corner_qubits = {(set(node1.qubits) & set(node2.qubits)).pop()
                         for node1, node2 in itertools.combinations(self.boundary_nodes, 2)}

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


@dataclasses.dataclass()
class SquarePlotter(Plotter2D):
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

    def postprocess_primary_node_layout(
            self, points: list[npt.NDArray[np.float64]], qubit_to_pointpos: dict[int, int],
    ) -> list[npt.NDArray[np.float64]]:
        """Move qubits at the outside more outward, to form an even plane at each boundary."""
        qubit_to_point = {qubit: points[pointpos] for qubit, pointpos in qubit_to_pointpos.items()}
        corner_qubits = {qubit for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 2}
        border_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 1}

        center = np.asarray([0.0, 0.0, 0.0])
        for point in points:
            center += point
        center = center / len(points)

        # move corner qubits more outward
        corner_factor = {
            4: 0.2,
            6: 0.13,
            8: 0.07,
            10: 0.05,
        }.get(self.distance, 0.03)
        for qubit in corner_qubits:
            coordinate = qubit_to_point[qubit]
            qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate + corner_factor * (coordinate - center)

        # move all points at a border to the line of their corner qubits.
        for qubit in border_qubits:
            coordinate = qubit_to_point[qubit]
            boundary_node = border_qubits[qubit][0]
            relevant_corner_qubits = list(corner_qubits & set(boundary_node.qubits))
            if len(relevant_corner_qubits) != 2:
                raise RuntimeError(relevant_corner_qubits)
            corner_qubit1 = qubit_to_point[relevant_corner_qubits[0]]
            corner_qubit2 = qubit_to_point[relevant_corner_qubits[1]]
            boundary_edge_center = (corner_qubit1 + corner_qubit2) / 2
            coordinate = project_to_line([corner_qubit1, corner_qubit2], coordinate)
            factor = self._primary_boundary_edge_factor(coordinate, boundary_edge_center)
            coordinate = coordinate + factor * (coordinate - boundary_edge_center)
            qubit_to_point[qubit] = points[qubit_to_pointpos[qubit]] = coordinate

        return points

    @staticmethod
    def _layout_dual_nodes_factor(distance: int) -> Optional[float]:
        return {
            4: 1.2,
            6: 0.96,
        }.get(distance)

    def preprocess_dual_node_layout(self, primary_mesh: pyvista.PolyData):
        lines = reconvert_faces(primary_mesh.lines)
        faces = reconvert_faces(primary_mesh.faces)
        border_qubits = {qubit: nodes for qubit, nodes in self.qubit_to_boundaries.items() if len(nodes) == 1}

        ret: dict[int, npt.NDArray[np.float64]] = {}
        for dg_index, face in zip(primary_mesh.cell_data['pyvista_indices'][len(lines):], faces):
            center = np.asarray([0.0, 0.0, 0.0])
            divisor = len(face)
            for pos in face:
                center += primary_mesh.points[pos]
                # align the center of hexagonal faces correctly
                if primary_mesh.point_data['qubits'][pos] in border_qubits and len(face) == 6:
                    center += 2 * primary_mesh.points[pos]
                    divisor += 2
            center = center / divisor
            ret[dg_index] = center

        return ret
