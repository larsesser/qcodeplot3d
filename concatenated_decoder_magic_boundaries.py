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

    x_stabilizer_by_color = collections.defaultdict(list)
    for stabilizer in x_stabilizer:
        x_stabilizer_by_color[stabilizer.color].append(stabilizer)
    print(f"\nx stabilizers:")
    for color, stabilizers in x_stabilizer_by_color.items():
        print(f"  color {color.name}: {count_independent(stabilizers)}/{len(stabilizers)}")

    z_stabilizer_by_color = collections.defaultdict(list)
    for stabilizer in z_stabilizer:
        z_stabilizer_by_color[stabilizer.color].append(stabilizer)
    print(f"\nz stabilizers:")
    for color, stabilizers in z_stabilizer_by_color.items():
        included_qubits = set()
        for stabilizer in stabilizers:
            included_qubits.update(stabilizer.qubits)
        print(
            f"  color {color.name}: {count_independent(stabilizers)}/{len(stabilizers)}, covered qubits {len(included_qubits)}")

    odd_stabilizers = [stabilizer for stabilizer in stabilizers if len(stabilizer.qubits) % 2 == 1]
    odd_stabilizers_lenghts = collections.Counter([len(stabilizer.qubits) for stabilizer in odd_stabilizers])
    print(f"\nOdd stabilizers: {len(odd_stabilizers)}")
    if odd_stabilizers:
        print("  " + ", ".join(f"length {length}: {count}" for length, count in odd_stabilizers_lenghts.most_common()))

    n = stabilizers[0].length
    k = n - num_independent
    print(f"\nn: {n}, k: {k}, d: {distance}")


def plot_graphs(graph: rustworkx.PyGraph, distance: int):
    plotter = Plotter3D(graph, distance=distance)

    plotter.show_debug_mesh(plotter.construct_debug_mesh(graph), show_labels=False, exclude_boundaries=False)
    plotter.show_dual_mesh(show_labels=False, explode_factor=0, exclude_boundaries=False)
    plotter.show_primary_mesh(explode_factor=0.4)

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

d=5
graph = tetrahedron_3d_dual_graph(d)
plotter = TetrahedronPlotter(graph, distance=d)
plotter.plot_primary_mesh()

exit()

def highlighted_nodes(graph: rustworkx.PyGraph, qubits: list[int]) -> list["DualGraphNode"]:
    return [node for node in graph.nodes() if len(set(node.qubits) & set(qubits)) % 2 and not node.is_boundary]

def get_syndrome(dual_graph: rustworkx.PyGraph, qubits: list[int]) -> Syndrome:
    return Syndrome({
        node.stabilizer: SyndromeValue(bool(len(set(node.qubits) & set(qubits)) % 2))
        for node in dual_graph.nodes() if node.is_stabilizer
    })

for d in [7]:
    graph = tetrahedron_3d_dual_graph(d)
    decoder = ConcatenatedDecoder(Kind.x, [Color.red, Color.blue, Color.green, Color.yellow], graph)
    plotter = TetrahedronPlotter(graph, distance=d)
    _calculate_edge_weights(graph)
    dg_coordinates = plotter.layout_dual_nodes(plotter.construct_primary_mesh())
    primary_coordinates = plotter.get_qubit_coordinates(plotter.construct_primary_mesh())
    # plotter.plot_primary_mesh()
    # plotter.plot_primary_mesh(highlighted_qubits=[1, 2, 3, 6, 7, 8, 9, 13, 14, 18, 22, 24, 25, 28, 36, 37, 38, 45, 50, 52, 53, 55, 59, 60, 63, 68, 70, 71, 72, 73, 74, 76, 77, 80, 82, 83, 88, 91, 95, 96, 103, 104, 105, 106, 107, 108, 109, 111, 113, 114, 115, 122, 123, 127, 133, 134, 135, 138, 140, 142, 143, 144, 146, 151, 152, 154, 155, 158, 160, 162, 164, 165, 168, 170, 172, 173])
    # plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(graph, coordinates=dg_coordinates))

    r_colors = [Color.green, Color.red]
    r_graph = decoder.restricted_graph(r_colors)
    index_map = {node.dg_node.index: node.index for node in r_graph.nodes()}
    # use 3D coordinates from dual graph layout, so its more clear which node corresponds to which in the visualization
    rg_coordinates = {index_map[index]: coordinate for index, coordinate in dg_coordinates.items() if index in index_map}
    # plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(r_graph, coordinates=rg_coordinates), show_qubit_labels=True)

    errors_on_qubits = [1, 75, 150]
    syndrome = get_syndrome(graph, errors_on_qubits)
    r_matching = decoder._decode_r_graph(syndrome, r_colors, propagate_weight=True)
    r_matching_edges = _matching2edges(r_matching, r_graph)
    r_nodes = highlighted_nodes(r_graph, errors_on_qubits)
    plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(r_graph, coordinates=rg_coordinates, highlighted_nodes=r_nodes), highlighted_qubits=errors_on_qubits, highlighted_edges=r_matching_edges, show_edge_weights=True)

    mc3_color = Color.blue
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
    mc3_nodes = highlighted_nodes(mc3_graph, errors_on_qubits)
    plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(mc3_graph, coordinates=mc3_coordinates, highlighted_nodes=mc3_nodes), highlighted_qubits=errors_on_qubits, highlighted_edges=mc3_matching_edges, show_edge_weights=True)

    mc4_color = Color.yellow
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

    mc4_matching = decoder._decode_mc4_graph(syndrome, r_colors, mc3_color, mc4_color, mc3_matchting)
    mc4_matching_edges = _matching2edges(mc4_matching, mc4_graph)
    mc4_nodes = highlighted_nodes(mc4_graph, errors_on_qubits)
    plotter.plot_debug_primary_mesh(plotter.construct_debug_mesh(mc4_graph, coordinates=mc4_coordinates, highlighted_nodes=mc4_nodes), highlighted_qubits=errors_on_qubits, highlighted_edges=mc4_matching_edges)

    # print([edge.qubit for edge in mc4_matching_edges])
    # print(decoder.decode(syndrome))
    # print(decoder.decode(syndrome, return_all_corrections=True))

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
