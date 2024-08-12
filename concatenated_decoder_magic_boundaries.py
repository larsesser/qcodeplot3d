import collections

import rustworkx
from rustworkx.visualization import graphviz_draw

from framework.cc_2d.construction import rectangular_2d_dual_graph, square_2d_dual_graph
from framework.cc_3d.base import GraphEdge, GraphNode
from framework.cc_3d.construction import (
    construct_x_dual_graph,
    cubic_3d_dual_graph,
    tetrahedron_3d_dual_graph,
)
from framework.cc_3d.decoder import ConcatenatedDecoder
from framework.cc_3d.plotter import Plotter3D
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


def print_stats(graph: rustworkx.PyGraph, distance: int):
    x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]
    z_stabilizer: list[Stabilizer] = [edge.stabilizer for edge in graph.edges() if edge.is_stabilizer]
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


def plot_graphs(graph: rustworkx.PyGraph):
    plotter = Plotter3D(graph)

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
    restricted_graph = decoder.restricted_graph([Color.red, Color.blue])
    mc3_graph = decoder.mc3_graph([Color.red, Color.blue], Color.green)
    mc4_graph = decoder.mc4_graph([Color.red, Color.blue], Color.green, Color.yellow)

    qubits = graph.nodes()[0].all_qubits
    x_stabilizer: list[Stabilizer] = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]

    # emulate no qubit errors
    print("\nEmulate single-qubit errors")
    results = decoder.decode(Syndrome({stabilizer: SyndromeValue(False) for stabilizer in x_stabilizer}), return_all_corrections=True)
    if any(result != [] for result in results):
        print(None, results)
    # emulate single-qubit errors
    for qubit in qubits:
        true_stabilizer = [stabilizer for stabilizer in x_stabilizer if qubit in stabilizer.qubits]
        syndrome = Syndrome({stabilizer: SyndromeValue(stabilizer in true_stabilizer) for stabilizer in x_stabilizer})
        results = decoder.decode(syndrome, return_all_corrections=True)
        if any(result != [qubit] for result in results):
            print(qubit, results)


d = 4
graph = cubic_3d_dual_graph(d)

print_stats(graph, d)
basic_decoder_test(graph)
plot_graphs(graph)

exit()

x_dual_graph = construct_x_dual_graph(graph)

# to run the following code, disable the NotImplemented Check in the decoder class (by removing the code)
decoder = ConcatenatedDecoder(Kind.x, [Color.rb, Color.rg, Color.ry, Color.bg, Color.by, Color.gy], x_dual_graph)
plotter = Plotter3D(x_dual_graph)

# for pairs of mutual exclusive two-color-colors, there is a 1:1 mapping from edges to qubits
# -> so, we can decode the syndrome by applying a MWPM approach to this graph directly
for colors in [[Color.rb, Color.gy], [Color.rg, Color.by], [Color.ry, Color.bg]]:
    restricted_graph = decoder.restricted_graph(colors)
    if len(restricted_graph.edges()) != len(graph.nodes()[0].all_qubits):
        print(f"{len(restricted_graph.edges())} edges vs {len(graph.nodes()[0].all_qubits)} qubits")
    if any(len(edge.qubits) != 1 for edge in restricted_graph.edges()):
        print("No 1:1 mapping from edges to qubits.")

    debug_mesh = plotter.construct_debug_mesh(restricted_graph)
    plotter.show_debug_mesh(debug_mesh, show_labels=False, exclude_boundaries=False)

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


exit()

decoder = ConcatenatedDecoder(Kind.x, [Color.red, Color.green, Color.yellow, Color.blue], construct_dual_graph())

graphviz_draw(decoder.dual_graph, node_attr_fn, edge_attr_fn, filename="mb_dualgraph.png", method="sfdp")
# https://stackoverflow.com/questions/14662618/is-there-a-3d-version-of-graphviz
exit()
# graphviz_draw(decoder.dual_graph, node_attr_fn, edge_attr_fn, filename="mb_dualgraph.vrml", method="sfdp", image_type="vrml", graph_attr={"dimen": "3"})
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow]), node_attr_fn, edge_attr_fn, filename="mb_restricted_gy.png", method="sfdp")
graphviz_draw(decoder.restricted_graph([Color.green, Color.yellow, Color.blue]), node_attr_fn, edge_attr_fn, filename="mb_restricted_gyb.png", method="sfdp")
graphviz_draw(decoder.mc3_graph([Color.green, Color.yellow], Color.blue), node_attr_fn, edge_attr_fn, filename="mb_monochrom_gy_b.png", method="sfdp")
graphviz_draw(decoder.mc4_graph([Color.green, Color.yellow], Color.blue, Color.red), node_attr_fn, edge_attr_fn, filename="mb_monochrom_gy_b_r.png", method="sfdp")
