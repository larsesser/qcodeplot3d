import collections

import rustworkx
from rustworkx.visualization import graphviz_draw

from framework.base import GraphEdge, GraphNode
from framework.cc_2d.construction import rectangular_2d_dual_graph, square_2d_dual_graph
from framework.cc_3d.construction import construct_x_dual_graph, cubic_3d_dual_graph, tetrahedron_d5_dual_graph, tetrahedron_3d_dual_graph, construct_tetrahedron_logicals
from framework.cc_3d.decoder import ConcatenatedDecoder, SubsetDecoder, _calculate_edge_weights, _adjust_edge_weights_with_syndrome, _matching2edges
from framework.cc_3d.plotter import CubicPlotter, TetrahedronPlotter
from framework.construction import construct_restricted_graph
from framework.stabilizers import (
    Color,
    Operator,
    Stabilizer,
    check_stabilizers,
    check_xj,
    check_z,
    count_independent,
)
from framework.syndrome import Syndrome, SyndromeValue
from framework.util import Kind


def node_attr_fn(node: GraphNode):
    label = f"{node.index}"
    if node.title:
        label = f"{node.title}"
    if node.is_boundary:
        label += " B"
    attr_dict = {
        "style": "filled",
        "shape": "point",
        "label": label,
    }
    if hasattr(node, "color"):
        attr_dict["color"] = node.color.name
        attr_dict["fill_color"] = node.color.name
    return attr_dict


def edge_attr_fn(edge: GraphEdge):
    attr_dict = {
    }
    return attr_dict


def print_stats(graph: rustworkx.PyGraph, distance: int, dimension: int = 3):
    x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]
    if dimension == 2:
        z_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]
    elif dimension == 3:
        z_stabilizer: list[Stabilizer] = [edge.stabilizer for edge in graph.edges() if edge.is_stabilizer]
    else:
        raise NotImplementedError
    stabilizers: list[Stabilizer] = [*x_stabilizer, *z_stabilizer]

    independent_x = count_independent(x_stabilizer)
    independent_z = count_independent(z_stabilizer)
    num_independent = independent_x + independent_z
    print(f"Stabilizers: {num_independent}/{len(stabilizers)}")
    print(f"  x stabilizer: {independent_x}/{len(x_stabilizer)}")
    print(f"  z stabilizer: {independent_z}/{len(z_stabilizer)}")

    #x_stabilizer_by_color = collections.defaultdict(list)
    #for stabilizer in x_stabilizer:
    #    x_stabilizer_by_color[stabilizer.color].append(stabilizer)
    #print(f"\nx stabilizers:")
    #for color, stabilizers in x_stabilizer_by_color.items():
    #    print(f"  color {color.name}: {count_independent(stabilizers)}/{len(stabilizers)}")
#
    #z_stabilizer_by_color = collections.defaultdict(list)
    #for stabilizer in z_stabilizer:
    #    z_stabilizer_by_color[stabilizer.color].append(stabilizer)
    #print(f"\nz stabilizers:")
    #for color, stabilizers in z_stabilizer_by_color.items():
    #    included_qubits = set()
    #    for stabilizer in stabilizers:
    #        included_qubits.update(stabilizer.qubits)
    #    print(
    #        f"  color {color.name}: {count_independent(stabilizers)}/{len(stabilizers)}, covered qubits {len(included_qubits)}")
#
    #odd_stabilizers = [stabilizer for stabilizer in stabilizers if len(stabilizer.qubits) % 2 == 1]
    #odd_stabilizers_lenghts = collections.Counter([len(stabilizer.qubits) for stabilizer in odd_stabilizers])
    #print(f"\nOdd stabilizers: {len(odd_stabilizers)}")
    #if odd_stabilizers:
    #    print("  " + ", ".join(f"length {length}: {count}" for length, count in odd_stabilizers_lenghts.most_common()))

    n = stabilizers[0].length
    k = n - num_independent
    print(f"n: {n}, k: {k}, d: {distance}\n")


def plot_graphs(graph: rustworkx.PyGraph, distance: int):
    plotter = Plotter3D(graph, distance=distance)

    plotter.show_debug_mesh(plotter.construct_debug_mesh(graph), show_labels=False, exclude_boundaries=False)
    # plotter.show_dual_mesh(show_labels=False, explode_factor=0, exclude_boundaries=False)
    plotter.show_primary_mesh()

    decoder = ConcatenatedDecoder(Kind.x, [Color.red, Color.blue, Color.green, Color.yellow], graph)
    restricted_graph = decoder.restricted_graph([Color.red, Color.blue])
    mc3_graph = decoder.mc3_graph([Color.red, Color.blue], Color.green)
    mc4_graph = decoder.mc4_graph([Color.red, Color.blue], Color.green, Color.yellow)

    for g in [restricted_graph, mc3_graph, mc4_graph]:
        debug_mesh = plotter.construct_debug_mesh(g)
        plotter.show_debug_mesh(debug_mesh, show_labels=False, exclude_boundaries=False)


def basic_decoder_test(graph: rustworkx.PyGraph):
    decoder = ConcatenatedDecoder(Kind.x, [Color.red, Color.blue, Color.green, Color.yellow], graph)

    qubits = graph.nodes()[0].all_qubits
    x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]

    # emulate no qubit errors
    print("\nEmulate single-qubit errors")
    results = decoder.decode(Syndrome({stabilizer: SyndromeValue(False) for stabilizer in x_stabilizer}), return_all_corrections=True)
    if any(result != [] for result in results.values()):
        print(None, results)
    # emulate single-qubit errors
    for qubit in qubits:
        true_stabilizer = [stabilizer for stabilizer in x_stabilizer if qubit in stabilizer.qubits]
        syndrome = Syndrome({stabilizer: SyndromeValue(stabilizer in true_stabilizer) for stabilizer in x_stabilizer})
        results = decoder.decode(syndrome, return_all_corrections=True)
        if any(result != [qubit] for result in results.values()):
            print(qubit, results)


def basic_x_decoder_test(graph: rustworkx.PyGraph):
    decoder = SubsetDecoder(Kind.z, [Color.rb, Color.gy, Color.rg, Color.by, Color.ry, Color.bg], graph)

    qubits = graph.nodes()[0].all_qubits
    z_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]

    # emulate no qubit errors
    print("\nEmulate single-qubit errors")
    results = decoder.decode(Syndrome({stabilizer: SyndromeValue(False) for stabilizer in z_stabilizer}), return_all_corrections=True)
    if any(result != [] for result in results.values()):
        print(None, results)
    # emulate single-qubit errors
    for qubit in qubits:
        true_stabilizer = [stabilizer for stabilizer in z_stabilizer if qubit in stabilizer.qubits]
        syndrome = Syndrome({stabilizer: SyndromeValue(stabilizer in true_stabilizer) for stabilizer in z_stabilizer})
        results = decoder.decode(syndrome, return_all_corrections=True)
        if any(result != [qubit] for result in results.values()):
            print(qubit, results)

# d=5
# graph = tetrahedron_3d_dual_graph(d)
# plotter = TetrahedronPlotter(graph, distance=d)
# plotter.plot_primary_mesh()
# exit()

def highlighted_nodes(graph: rustworkx.PyGraph, qubits: list[int]) -> list["DualGraphNode"]:
    return [node for node in graph.nodes() if len(set(node.qubits) & set(qubits)) % 2 and not node.is_boundary]

def get_endpoint_nodes(edges: list[GraphEdge]) -> list[GraphNode]:
    node_ids = set()
    for edge in edges:
        for node in [edge.node1, edge.node2]:
            if node.is_boundary:
                continue
            if node.id in node_ids:
                node_ids.remove(node.id)
            else:
                node_ids.add(node.id)
    return [node for edge in edges for node in (edge.node1, edge.node2) if node.id in node_ids]

def get_syndrome(dual_graph: rustworkx.PyGraph, qubits: list[int]) -> Syndrome:
    return Syndrome({
        node.stabilizer: SyndromeValue(bool(len(set(node.qubits) & set(qubits)) % 2))
        for node in dual_graph.nodes() if node.is_stabilizer
    })

d = 7
colors = [
    #([Color.red, Color.green], Color.blue, Color.yellow),
    #([Color.red, Color.green], Color.yellow, Color.blue),
    #([Color.red, Color.blue], Color.green, Color.yellow),
    ([Color.red, Color.blue], Color.yellow, Color.green),
    ([Color.red, Color.yellow], Color.blue, Color.green),
    ([Color.red, Color.yellow], Color.green, Color.blue),
    ([Color.green, Color.blue], Color.red, Color.yellow),
    ([Color.green, Color.blue], Color.yellow, Color.red),
    ([Color.green, Color.yellow], Color.blue, Color.red),
    ([Color.green, Color.yellow], Color.red, Color.blue),
    ([Color.blue, Color.yellow], Color.red, Color.green),
    ([Color.blue, Color.yellow], Color.green, Color.red),
]
for r_colors, mc3_color, mc4_color in colors:
    graph = tetrahedron_3d_dual_graph(d)
    decoder = ConcatenatedDecoder(Kind.x, [Color.red, Color.blue, Color.green, Color.yellow], graph)
    plotter = TetrahedronPlotter(graph, distance=d)
    _calculate_edge_weights(graph)
    dg_coordinates = plotter.layout_dual_nodes(plotter.construct_primary_mesh())
    primary_coordinates = plotter.get_qubit_coordinates(plotter.construct_primary_mesh())
    # plotter.plot_primary_mesh()
    # plotter.plot_primary_mesh(highlighted_qubits=[1, 2, 3, 6, 7, 8, 9, 13, 14, 18, 22, 24, 25, 28, 36, 37, 38, 45, 50, 52, 53, 55, 59, 60, 63, 68, 70, 71, 72, 73, 74, 76, 77, 80, 82, 83, 88, 91, 95, 96, 103, 104, 105, 106, 107, 108, 109, 111, 113, 114, 115, 122, 123, 127, 133, 134, 135, 138, 140, 142, 143, 144, 146, 151, 152, 154, 155, 158, 160, 162, 164, 165, 168, 170, 172, 173])
    # plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(graph, coordinates=dg_coordinates))

    r_graph = decoder.restricted_graph(r_colors)
    index_map = {node.dg_node.index: node.index for node in r_graph.nodes()}
    # use 3D coordinates from dual graph layout, so its more clear which node corresponds to which in the visualization
    rg_coordinates = {index_map[index]: coordinate for index, coordinate in dg_coordinates.items() if index in index_map}
    # plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(r_graph, coordinates=rg_coordinates), show_qubit_labels=True)

    # errors_on_qubits = [54, 99, 147]
    # errors_on_qubits = [1, 11, 47]
    # errors_on_qubits = [61, 75, 112]
    # errors_on_qubits = [6, 18, 126]
    # errors_on_qubits = [94, 98, 124]
    errors_on_qubits = [1, 26, 150]
    plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(graph, coordinates=dg_coordinates, highlighted_nodes=highlighted_nodes(graph, errors_on_qubits)),
                                    highlighted_qubits=errors_on_qubits)
    show_weights = False
    syndrome = get_syndrome(graph, errors_on_qubits)
    r_matching = decoder._decode_r_graph(syndrome, r_colors, propagate_weight=True)
    r_matching_edges = _matching2edges(r_matching, r_graph)
    r_nodes = highlighted_nodes(r_graph, errors_on_qubits)
    plotter.plot_debug_primary_mesh(
        plotter.construct_debug_mesh(r_graph, coordinates=rg_coordinates, highlighted_nodes=r_nodes),
        highlighted_qubits=errors_on_qubits, highlighted_edges=r_matching_edges, show_edge_weights=show_weights)

    mc3_graph = decoder.mc3_graph(r_colors, mc3_color)
    # use 3D coordinates from dual graph layout, so its more clear which node corresponds to which in the visualization
    mc3_coordinates = {}
    center: list[float] = sum(dg_coordinates.values()) / len(dg_coordinates)
    for node in mc3_graph.nodes():
        if node.rg_node is not None:
            mc3_coordinates[node.index] = dg_coordinates[node.rg_node.dg_node.index]
        elif node.rg_edge is not None:
            node1 = node.rg_edge.dg_edge.node1
            node2 = node.rg_edge.dg_edge.node2
            # position of node is on center of the face between the rg nodes
            face_qubits = set(node1.qubits) & set(node2.qubits)
            coordinate = sum(primary_coordinates[qubit] for qubit in face_qubits) / len(face_qubits)
            mc3_coordinates[node.index] = coordinate
        else:
            raise RuntimeError

    mc3_matchting = decoder._decode_mc3_graph(syndrome, r_colors, mc3_color, r_matching, propagate_weight=True)
    mc3_matching_edges = _matching2edges(mc3_matchting, mc3_graph)
    mc3_nodes = get_endpoint_nodes(mc3_matching_edges)
    plotter.plot_debug_primary_mesh(
        plotter.construct_debug_mesh(mc3_graph, coordinates=mc3_coordinates, highlighted_nodes=mc3_nodes),
        highlighted_qubits=errors_on_qubits, highlighted_edges=mc3_matching_edges, show_edge_weights=show_weights)

    mc4_graph = decoder.mc4_graph(r_colors, mc3_color, mc4_color)
    # use 3D coordinates from dual graph layout, so its more clear which node corresponds to which in the visualization
    mc4_coordinates = {}
    center: list[float] = sum(dg_coordinates.values()) / len(dg_coordinates)
    for node in mc4_graph.nodes():
        if node.dg_node is not None:
            mc4_coordinates[node.index] = dg_coordinates[node.dg_node.index]
        elif node.mc3_edge is not None:
            nodes = node.dg_nodes
            # position of node is on center of the edge between the rg nodes
            edge_qubits = set(nodes[0].qubits) & set(nodes[1].qubits) & set(nodes[2].qubits)
            coordinate = sum(primary_coordinates[qubit] for qubit in edge_qubits) / len(edge_qubits)
            mc4_coordinates[node.index] = coordinate
        else:
            raise RuntimeError

    mc4_matching = decoder._decode_mc4_graph(syndrome, r_colors, mc3_color, mc4_color, mc3_matchting, propagate_weight=True)
    mc4_matching_edges = _matching2edges(mc4_matching, mc4_graph)
    mc4_nodes = get_endpoint_nodes(mc4_matching_edges)
    plotter.plot_debug_primary_mesh(
        plotter.construct_debug_mesh(mc4_graph, coordinates=mc4_coordinates, highlighted_nodes=mc4_nodes),
        highlighted_qubits=errors_on_qubits, highlighted_edges=mc4_matching_edges, show_edge_weights=show_weights)

    # print([edge.qubit for edge in mc4_matching_edges])
    # print(decoder.decode(syndrome))
    # print(decoder.decode(syndrome, return_all_corrections=True))

exit()

d = 7
graph = tetrahedron_3d_dual_graph(d)

error_qubits = [[1, 26, 150], [1, 75, 150], [3, 47, 171], [5, 59, 134], [5, 99, 147], [6, 17, 126], [6, 108, 126], [7, 25, 116], [7, 45, 98], [8, 16, 89], [10, 17, 84], [14, 28, 145], [14, 61, 146], [14, 74, 90], [14, 80, 87], [16, 40, 133], [16, 96, 163], [16, 123, 175], [17, 18, 108], [17, 20, 172], [19, 35, 71], [19, 59, 170], [19, 134, 170], [19, 147, 170], [19, 160, 161], [20, 108, 172], [21, 59, 165], [21, 134, 165], [21, 147, 165], [24, 87, 125], [25, 53, 128], [25, 53, 131], [25, 53, 162], [26, 47, 77], [27, 32, 61], [27, 43, 61], [27, 61, 117], [27, 79, 145], [28, 32, 113], [28, 43, 113], [28, 80, 145], [28, 90, 145], [28, 113, 117], [29, 59, 148], [29, 99, 103], [32, 80, 92], [35, 39, 71], [35, 42, 50], [35, 71, 87], [35, 85, 152], [37, 47, 100], [37, 86, 150], [39, 74, 125], [39, 160, 161], [42, 47, 50], [45, 53, 118], [46, 50, 131], [47, 75, 77], [47, 85, 152], [50, 74, 131], [59, 140, 148], [61, 80, 146], [61, 90, 146], [73, 85, 131], [74, 85, 131], [79, 113, 146], [87, 160, 161], [90, 92, 105], [98, 124, 128], [98, 124, 131], [98, 124, 162], [99, 103, 140], [103, 134, 137], [103, 134, 155], [116, 118, 124], [126, 127, 164], [137, 147, 148], [141, 164, 172], [147, 148, 155]]
plotter = TetrahedronPlotter(graph, distance=d)
for qubits in error_qubits:
    plotter.plot_primary_mesh(highlighted_qubits=qubits, show_normal_qubits=False)


# for d in [3, 5, 7, 9]:
#     graph = tetrahedron_3d_dual_graph(d)
#     print(f"n: {len(graph[0].all_qubits)}, k: 1, d: {d}")
# print_stats(graph, d, dimension=3)
# construct_tetrahedron_logicals(graph)
# plotter = TetrahedronPlotter(graph, distance=d)
# plotter.plot_debug_mesh(plotter.construct_debug_mesh(graph))
# dual_coordinates = plotter.layout_dual_nodes(plotter.construct_primary_mesh())
# plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(graph, coordinates=dual_coordinates))
# plotter.plot_primary_mesh()

exit()

print(len(graph.nodes()[0].all_qubits))
print(count_independent([node.stabilizer for node in graph.nodes() if node.stabilizer]) + count_independent([edge.stabilizer for edge in graph.edges() if edge.stabilizer]))

exit()
# print(g.nodes())
# print(g.edge_list())


graph = tetrahedron_3d_dual_graph(3)
plotter = TetrahedronPlotter(graph, distance=d)
plotter.plot_debug_mesh(plotter.construct_debug_mesh(g))

exit()

dual_coordinates = plotter.layout_dual_nodes(plotter.construct_primary_mesh())
mesh = plotter.construct_debug_mesh(graph, coordinates=dual_coordinates, exclude_boundaries=True, include_edges_between_boundaries=False, highlighted_nodes=graph.nodes())
plotter.plot_debug_mesh(mesh, show_labels=True)
# plotter.plot_primary_mesh(show_normal_qubits=True, line_width=3)
# plotter.plot_debug_primary_mesh(mesh, show_normal_qubits=False, primary_line_width=3, mesh_line_width=1)
exit()

d = 6
graph = cubic_3d_dual_graph(d)
plotter = CubicPlotter(graph, distance=d)
dual_coordinates = plotter.layout_dual_nodes(plotter.construct_primary_mesh())
plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(graph, coordinates=dual_coordinates))
exit()

print_stats(graph, d, dimension=2)

exit()

basic_decoder_test(graph)
plot_graphs(graph, d)

exit()

x_dual_graph = construct_x_dual_graph(graph)

plotter = Plotter3D(x_dual_graph, distance=d)

# for pairs of mutual exclusive two-color-colors, there is a 1:1 mapping from edges to qubits
# -> so, we can decode the syndrome by applying a MWPM approach to this graph directly
for colors in [[Color.rb, Color.gy], [Color.rg, Color.by], [Color.ry, Color.bg]]:
    restricted_graph = construct_restricted_graph(x_dual_graph, colors)
    if len(restricted_graph.edges()) != len(graph.nodes()[0].all_qubits):
        print(f"{len(restricted_graph.edges())} edges vs {len(graph.nodes()[0].all_qubits)} qubits")
    if any(len(edge.qubits) != 1 for edge in restricted_graph.edges()):
        print("No 1:1 mapping from edges to qubits.")

    debug_mesh = plotter.construct_debug_mesh(restricted_graph)
    plotter.show_debug_mesh(debug_mesh, show_labels=False, exclude_boundaries=False)

basic_x_decoder_test(x_dual_graph)

exit()

d = 4
graph = square_2d_dual_graph(d)

x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.stabilizer]
z_stabilizer: list[Stabilizer] = [Stabilizer(s.length, s.color, z_positions=s.x) for s in x_stabilizer]
stabilizers: list[Stabilizer] = [*x_stabilizer, *z_stabilizer]
check_stabilizers(stabilizers)

logical_1_pos = [node for node in graph.nodes() if node.title == "bottom"][0].qubits
z_1 = Operator(length=stabilizers[0].length, z_positions=logical_1_pos)
x_2 = Operator(length=stabilizers[0].length, x_positions=logical_1_pos)

logical_2_pos = [node for node in graph.nodes() if node.title == "left"][0].qubits
z_2 = Operator(length=stabilizers[0].length, z_positions=logical_2_pos)
x_1 = Operator(length=stabilizers[0].length, x_positions=logical_2_pos)

check_z([z_1, z_2], stabilizers)
check_xj(x_1, z_1, [z_2], stabilizers)
check_xj(x_2, z_2, [z_1], stabilizers)

n = stabilizers[0].length
k = n - len(stabilizers)
print(f"n: {n}, k: {k}, d: {d}")

graphviz_draw(graph, node_attr_fn, filename="2D square d=4.png", method="fdp")


exit()

graph = rectangular_2d_dual_graph(5)

x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.stabilizer]
z_stabilizer: list[Stabilizer] = [Stabilizer(s.length, s.color, z_positions=s.x) for s in x_stabilizer]
stabilizers = [*x_stabilizer, *z_stabilizer]
check_stabilizers(stabilizers)

logical_1_pos = [node for node in graph.nodes() if node.title == "bottom"][0].qubits
z_1 = Operator(length=stabilizers[0].length, z_positions=logical_1_pos)
x_1 = Operator(length=stabilizers[0].length, x_positions=logical_1_pos)

logical_2_pos = [node for node in graph.nodes() if node.title == "top"][0].qubits
z_2 = Operator(length=stabilizers[0].length, z_positions=logical_2_pos)
x_2 = Operator(length=stabilizers[0].length, x_positions=logical_2_pos)

check_z([z_1, z_2], stabilizers)
check_xj(x_1, z_1, [z_2], stabilizers)
check_xj(x_2, z_2, [z_1], stabilizers)

graphviz_draw(graph, node_attr_fn, filename="2D rectangular d=3.png", method="fdp")
