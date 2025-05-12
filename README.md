# qCodePlot3D

qCodePlot3D provides 3D visualizations of 2D and 3D quantum color codes, making it easy to inspect the code and visualize syndromes of error configurations.

## Available Quantum Codes

The following quantum codes are currently implemented:
- 2D square color code
- 2D triangular color code
- 3D cubic color code
- 3D tetrahedral color code

The package provides an interface to extend it to further quantum codes.
Contributions of such implementations to the package are appreciated.


## Usage

The package comes with a minimal GUI to construct and plot the predefined quantum codes.
Start the GUI by calling
```
python3 -m qcodeplot3d
```

Alternatively, the package can be used to script the creation of figures.
A small example snipped might look as following:

```
import pathlib
from qcodeplot3d.cc_3d import TetrahedronPlotter, tetrahedron_3d_dual_graph

THIS_DIR = pathlib.Path(__file__).parent

distance = 5
dual_graph = tetrahedron_3d_dual_graph(distance)
plotter = TetrahedronPlotter(dual_graph, distance=distance)
# describes the position of the camera in the 3D visualization
CPOS = [(333.44920594948127, -92.62132121918515, 221.35429214703458),
        (-1.365808884682137, 1.7594735760392801, -12.072404824078472),
        (0.5292985009453346, 0.7030498941163704, -0.4749357254197479)]
plotter.plot_primary_mesh(
    camera_position=CPOS,
    filename= THIS_DIR / "tetrahedron-d=5-1-primary"
)
dual_graph_mesh = plotter.construct_dual_mesh(
    dual_graph,
    use_edges_colors=True,
    highlighted_edges=dual_graph.edges(),
    highlighted_nodes=dual_graph.nodes()
)
plotter.plot_debug_primary_mesh(
    dual_graph_mesh,
    camera_position=CPOS,
    transparent_faces=True,
    filename= THIS_DIR / "tetrahedron-d=5-2-both"
)
```
