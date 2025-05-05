from concurrent.futures import Executor, Future, ProcessPoolExecutor
from enum import Enum
from tkinter import *
from tkinter import ttk
from typing import Callable, Optional, Type

import pyvista
import rustworkx

from framework.cc_2d import SquarePlotter, rectangular_2d_dual_graph, square_2d_dual_graph
from framework.cc_3d import CubicPlotter, TetrahedronPlotter, cubic_3d_dual_graph, tetrahedron_3d_dual_graph
from framework.common.plotter import Plotter


class CodeTypes(Enum):
    # 2D codes
    rectangular = "Rectangular"
    square = "Square"

    # 3D codes
    cubic = "Cubic"
    tetrahedral = "Tetrahedral"

    @property
    def plotter_class(self) -> Type[Plotter]:
        return {
            self.rectangular: ...,
            self.square: SquarePlotter,
            self.cubic: CubicPlotter,
            self.tetrahedral: TetrahedronPlotter,
        }[self]

    @property
    def graph_function(self) -> Callable:
        return {
            self.rectangular: rectangular_2d_dual_graph,
            self.square: square_2d_dual_graph,
            self.cubic: cubic_3d_dual_graph,
            self.tetrahedral: tetrahedron_3d_dual_graph,
        }[self]

    @property
    def has_odd_distance(self) -> bool:
        return self in {self.tetrahedral}

    @property
    def has_even_distance(self) -> bool:
        return not self.has_odd_distance

    @classmethod
    def default_2d_code(cls) -> str:
        return cls.rectangular.value

    @classmethod
    def get_2d_codes(cls) -> list[str]:
        return [cls.rectangular.value]

    @classmethod
    def default_3d_code(cls) -> str:
        return cls.tetrahedral.value

    @classmethod
    def get_3d_codes(cls) -> list[str]:
        return [cls.cubic.value, cls.tetrahedral.value]


def change_state(elem: ttk.Widget, state: str) -> None:
    # we use Combobox as dropdown and do not support custom entries
    if isinstance(elem, ttk.Combobox) and state == "normal":
        state = "readonly"
    elem.configure(state=state)


class CodeConfig:
    root: Tk
    pool: Executor

    _dimension: IntVar
    _codetype: StringVar
    _distance: IntVar
    _distance_error_msg: StringVar

    # these are always in sync
    dimension: int = None
    distance: int = None
    codetype: CodeTypes = None
    dual_graph: rustworkx.PyGraph = None

    def __init__(self, root: Tk, pool: Executor):
        self.root = root
        self.pool = pool
        self._dimension = IntVar(value=3)
        self._codetype = StringVar(value=CodeTypes.default_3d_code())
        self._distance = IntVar(value=3)
        self._distance_error_msg = StringVar()

    def _create_distance_validator(self, submit: ttk.Button) -> Callable:
        def validator(new_value: str, operation: str) -> bool:
            self._distance_error_msg.set('')
            if new_value == "":
                change_state(submit, state="disabled")
                return True
            valid = True
            if not new_value.isdigit():
                self._distance_error_msg.set("Only digits are allowed.")
                valid = False
            elif operation in {'focusout', 'focusin', 'forced'}:
                is_even = int(new_value) % 2 == 0
                ct = CodeTypes(self._codetype.get())
                valid = False
                if not is_even and ct.has_even_distance:
                    self._distance_error_msg.set("Code requires even distance.")
                elif is_even and ct.has_odd_distance:
                    self._distance_error_msg.set("Code requires odd distance.")
                elif int(new_value) == 0 and ct.has_even_distance:
                    self._distance_error_msg.set("Minimal distance: 2")
                elif int(new_value) == 1 and ct.has_odd_distance:
                    self._distance_error_msg.set("Minimal distance: 3")
                else:
                    valid = True
            if valid:
                change_state(submit, state="normal")
            else:
                change_state(submit, state="disabled")
            return valid
        return validator

    def _create_dimension_callback(self, dimension: int, code_type: ttk.Combobox, distance: ttk.Entry) -> Callable:
        def callback() -> None:
            if dimension == 2:
                self._codetype.set(CodeTypes.default_2d_code())
                code_type.configure(values=CodeTypes.get_2d_codes())
            elif dimension == 3:
                self._codetype.set(CodeTypes.default_3d_code())
                code_type.configure(values=CodeTypes.get_3d_codes())
            else:
                raise NotImplementedError
            distance.validate()
        return callback

    def _create_submit_command(self, all_ttk: list[ttk.Widget], progressbar: ttk.Progressbar) -> Callable:
        def command() -> None:
            def callback(f: Future) -> None:
                self.dual_graph = f.result()
                progressbar.stop()
                self.dimension = self._dimension.get()
                self.codetype = CodeTypes(self._codetype.get())
                self.distance = self._distance.get()
                for elem in all_ttk:
                    change_state(elem, state="normal")
                self.root.event_generate("<<DualGraphCreationFinished>>")

            progressbar.start()
            self.root.event_generate("<<DualGraphCreationStarted>>")
            for elem_ in all_ttk:
                change_state(elem_, state="disabled")
            f_: Future = self.pool.submit(CodeTypes(self._codetype.get()).graph_function, self._distance.get())
            f_.add_done_callback(callback)
        return command

    def create_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, borderwidth=5, relief="ridge", padding=(3, 3, 12, 12), text="Code Config")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        two_d = ttk.Radiobutton(frame)
        three_d = ttk.Radiobutton(frame)
        code_type = ttk.Combobox(frame)
        distance = ttk.Entry(frame)
        submit_button = ttk.Button(frame)
        all_ttk = [two_d, three_d, code_type, distance, submit_button]
        progressbar = ttk.Progressbar(frame)

        # dimension radio buttons
        dimension_label = ttk.Label(frame, text='Dimension')
        two_d.configure(variable=self._dimension, value=2, text="2D", command=self._create_dimension_callback(2, code_type, distance))
        three_d.configure(variable=self._dimension, value=3, text="3D", command=self._create_dimension_callback(3, code_type, distance))
        dimension_label.grid(row=0, column=0)
        two_d.grid(row=0, column=1)
        three_d.grid(row=0, column=2)

        # code type dropdown
        code_type_label = ttk.Label(frame, text='Type')
        code_type.configure(textvariable=self._codetype, values=CodeTypes.get_3d_codes())
        code_type.bind("<<ComboboxSelected>>", lambda _: distance.validate())
        # only allow selection of predefined values
        code_type.state(["readonly"])
        code_type_label.grid(row=10, column=0)
        code_type.grid(row=10, column=1, columnspan=2, sticky="ew")

        # distance entry
        distance_label = ttk.Label(frame, text="Distance")
        validate_distance_wrapper = (frame.register(self._create_distance_validator(submit_button)), '%P', '%V')
        distance.configure(textvariable=self._distance, validate='all', validatecommand=validate_distance_wrapper)
        distance_msg = ttk.Label(frame, font='TkSmallCaptionFont', foreground='red', textvariable=self._distance_error_msg)
        distance_label.grid(row=20, column=0)
        distance.grid(row=20, column=1, columnspan=2, sticky="ew")
        distance_msg.grid(row=21, column=1, columnspan=2, padx=5, pady=5, sticky='w')

        # submit button, progress bar
        submit_button.configure(text="Build", command=self._create_submit_command(all_ttk, progressbar))
        submit_button.grid(row=30, column=2, sticky="es")
        progressbar.configure(orient="horizontal", mode="indeterminate")
        progressbar.grid(row=30, column=0, sticky="ws")
        frame.rowconfigure(30, weight=1)

        return frame


class PlotterConfig:
    root: Tk
    pool: Executor
    code_config: CodeConfig

    # create dual mesh
    use_edge_color: BooleanVar
    edges_between_boundaries: BooleanVar
    exclude_boundaries: BooleanVar

    # plot dual mesh
    dm_show_labels: BooleanVar

    plotter: Plotter = None
    dual_graph_mesh: pyvista.PolyData = None

    def __init__(self, root: Tk, pool: Executor, code_config: CodeConfig) -> None:
        self.root = root
        self.pool = pool
        self.code_config = code_config

        self.use_edge_color = BooleanVar(value=False)
        self.edges_between_boundaries = BooleanVar(value=True)
        self.exclude_boundaries = BooleanVar(value=False)

        self.dm_show_labels = BooleanVar(value=False)

        self.root.bind("<<DualGraphCreationFinished>>", self._update_plotter, add="+")

    @property
    def plotter_class(self) -> Optional[Type[Plotter]]:
        if self.code_config.codetype:
            return self.code_config.codetype.plotter_class
        return None

    def _update_plotter(self, _) -> None:
        cfg = self.code_config
        self.plotter = self.plotter_class(dual_graph=cfg.dual_graph, distance=cfg.distance)

    @staticmethod
    def _create_state_change_callback(*args, state: str) -> Callable:
        def callback(*_) -> None:
            for element in args:
                change_state(element, state=state)
        return callback

    def _create_dualmesh_create_command(self, all_ttk: list[ttk.Widget], progressbar: ttk.Progressbar) -> Callable:
        def command() -> None:
            def callback(f: Future) -> None:
                self.dual_graph_mesh = f.result()
                progressbar.stop()
                for elem in all_ttk:
                    change_state(elem, state="normal")
                self.root.event_generate("<<DualMeshCreationFinished>>")

            progressbar.start()
            for elem_ in all_ttk:
                change_state(elem_, state="disabled")
            self.root.event_generate("<<DualMeshCreationStarted>>")
            f_: Future = self.pool.submit(
                self.plotter.construct_dual_mesh, self.code_config.dual_graph,
                use_edges_colors=self.use_edge_color.get(),
                include_edges_between_boundaries=self.edges_between_boundaries.get(),
                exclude_boundaries=self.exclude_boundaries.get(),
                # highlight all nodes to make them more distinguishable
                highlighted_nodes=self.code_config.dual_graph.nodes(),
            )
            f_.add_done_callback(callback)
        return command

    def create_dual_config_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, borderwidth=5, relief="ridge", padding=(3, 3, 12, 12), text="Dual Graph Config")
        frame.configure()
        frame.columnconfigure(1, weight=1)

        use_edge_color = ttk.Checkbutton(frame)
        edges_between_boundaries = ttk.Checkbutton(frame)
        exclude_boundaries = ttk.Checkbutton(frame)
        # TODO: highlighted nodes, highlighted edges, mandatory qubits
        submit = ttk.Button(frame)
        all_ttk = [use_edge_color, edges_between_boundaries, exclude_boundaries, submit]
        progress_bar = ttk.Progressbar(frame)

        self.root.bind("<<DualGraphCreationStarted>>", self._create_state_change_callback(*all_ttk, state="disabled"), add="+")
        self.root.bind("<<DualGraphCreationFinished>>", self._create_state_change_callback(*all_ttk, state="normal"), add="+")
        for element in all_ttk:
            element.configure(state="disabled")

        use_edge_color.configure(variable=self.use_edge_color, text="Use colors of edges")
        use_edge_color.grid(row=0, column=1, sticky="w")

        exclude_boundaries.configure(variable=self.exclude_boundaries, text="Exclude boundary nodes")
        exclude_boundaries.grid(row=10, column=1, sticky="w")

        edges_between_boundaries.configure(variable=self.edges_between_boundaries, text="Show edges between boundaries")
        edges_between_boundaries.grid(row=20, column=1, sticky="w")

        submit.configure(text="Build", command=self._create_dualmesh_create_command(all_ttk, progress_bar))
        submit.grid(row=100, column=1, sticky="se")
        progress_bar.configure(orient="horizontal", mode="indeterminate")
        progress_bar.grid(row=100, column=0, sticky="sw")
        frame.rowconfigure(100, weight=1)

        return frame

    def _dualmesh_plot_command(self) -> None:
        self.pool.submit(self.plotter.plot_debug_mesh, self.dual_graph_mesh,
                         show_labels=self.dm_show_labels.get())

    def create_dual_plot_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, borderwidth=5, relief="ridge", padding=(3, 3, 12, 12), text="Plot Dual Graph")
        frame.configure()
        frame.columnconfigure(1, weight=1)

        show_labels = ttk.Checkbutton(frame)
        submit_button = ttk.Button(frame)
        all_ttk = [show_labels, submit_button]

        self.root.bind("<<DualGraphCreationStarted>>", self._create_state_change_callback(*all_ttk, state="disabled"), add="+")
        self.root.bind("<<DualMeshCreationStarted>>", self._create_state_change_callback(*all_ttk, state="disabled"), add="+")
        self.root.bind("<<DualMeshCreationFinished>>", self._create_state_change_callback(*all_ttk, state="normal"), add="+")
        for element in all_ttk:
            element.configure(state="disabled")

        show_labels.configure(variable=self.dm_show_labels, text="Show node labels")
        show_labels.grid(row=0, column=1, sticky="w")

        submit_button.configure(text="Plot", command=self._dualmesh_plot_command)
        submit_button.grid(row=10, column=1, sticky="se")
        frame.rowconfigure(10, weight=1)

        return frame


class MyGui:
    pool: Executor
    code_config: CodeConfig

    def __init__(self, root: Tk, pool: Executor) -> None:
        self.pool = pool
        root.title("...")

        content = ttk.Frame(root, padding=(3, 3, 12, 12))
        content.grid(row=0, column=0, sticky="nsew")

        self.code_config = CodeConfig(root, pool)
        code_config_frame = self.code_config.create_frame(content)
        code_config_frame.grid(row=0, column=0, sticky="nsew")

        self.plotter_config = PlotterConfig(root, pool, self.code_config)
        plotter_config_frame = self.plotter_config.create_dual_config_frame(content)
        plotter_config_frame.grid(row=0, column=10, sticky="nsew")
        plotter_dm_plot_frame = self.plotter_config.create_dual_plot_frame(content)
        plotter_dm_plot_frame.grid(row=10, column=0, sticky="nsew")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(10, weight=1)


thread_pool = ProcessPoolExecutor()
gui_root = Tk()
MyGui(gui_root, thread_pool)
gui_root.mainloop()
