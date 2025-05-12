import os
import signal
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from enum import Enum
from tkinter import BooleanVar, IntVar, StringVar, Tk, ttk
from typing import Callable, Optional

import psutil
import pyvista
import rustworkx

from qcodeplot3d.cc_2d import SquarePlotter, TriangularPlotter, square_2d_dual_graph, triangular_2d_dual_graph
from qcodeplot3d.cc_3d import CubicPlotter, TetrahedronPlotter, cubic_3d_dual_graph, tetrahedron_3d_dual_graph
from qcodeplot3d.common.graph import DualGraphNode
from qcodeplot3d.common.plotter import Plotter


class CodeTypes(Enum):
    # 2D codes
    triangular = "Triangular"
    square = "Square"

    # 3D codes
    cubic = "Cubic"
    tetrahedral = "Tetrahedral"

    @property
    def plotter_class(self) -> type[Plotter]:
        return {
            self.triangular: TriangularPlotter,
            self.square: SquarePlotter,
            self.cubic: CubicPlotter,
            self.tetrahedral: TetrahedronPlotter,
        }[self]

    @property
    def graph_function(self) -> Callable:
        return {
            self.triangular: triangular_2d_dual_graph,
            self.square: square_2d_dual_graph,
            self.cubic: cubic_3d_dual_graph,
            self.tetrahedral: tetrahedron_3d_dual_graph,
        }[self]

    @property
    def has_odd_distance(self) -> bool:
        return self in {self.triangular, self.tetrahedral}

    @property
    def has_even_distance(self) -> bool:
        return not self.has_odd_distance

    @classmethod
    def default_2d_code(cls) -> str:
        return cls.triangular.value

    @classmethod
    def get_2d_codes(cls) -> list[str]:
        return [cls.square.value, cls.triangular.value]

    @classmethod
    def default_3d_code(cls) -> str:
        return cls.tetrahedral.value

    @classmethod
    def get_3d_codes(cls) -> list[str]:
        return [cls.cubic.value, cls.tetrahedral.value]


def change_state(elem: ttk.Widget | list[ttk.Widget], state: str) -> None:
    # we use Combobox as dropdown and do not support custom entries
    if isinstance(elem, ttk.Combobox) and state == "normal":
        state = "readonly"
    if isinstance(elem, ttk.Widget):
        elem.configure(state=state)
    else:
        for e in elem:
            e.configure(state=state)


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

    @property
    def code_description(self) -> str:
        if self.codetype is None or self.dimension is None or self.distance is None:
            return ""
        return f"{self.dimension}D {self.codetype.value} Color Code, distance {self.distance}"

    def _create_distance_validator(self, error_store: StringVar, submit: ttk.Button) -> Callable:
        def validator(new_value: str, operation: str) -> bool:
            error_store.set('')
            if new_value == "":
                change_state(submit, state="disabled")
                return True
            if not new_value.isdigit():
                error_store.set("Only digits are allowed.")
                return False
            valid = True
            if operation in {'focusout', 'focusin', 'forced'}:
                is_even = int(new_value) % 2 == 0
                ct = CodeTypes(self._codetype.get())
                valid = False
                if not is_even and ct.has_even_distance:
                    error_store.set("Code requires even distance.")
                elif is_even and ct.has_odd_distance:
                    error_store.set("Code requires odd distance.")
                elif int(new_value) == 0 and ct.has_even_distance:
                    error_store.set("Minimal distance: 2")
                elif int(new_value) == 1 and ct.has_odd_distance:
                    error_store.set("Minimal distance: 3")
                else:
                    valid = True
            change_state(submit, state="normal" if valid else "disabled")
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
                change_state(all_ttk, state="normal")
                self.root.event_generate("<<DualGraphCreationFinished>>")

            progressbar.start()
            self.root.event_generate("<<DualGraphCreationStarted>>")
            change_state(all_ttk, state="disabled")
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
        validate_distance_wrapper = (frame.register(self._create_distance_validator(self._distance_error_msg, submit_button)), '%P', '%V')
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
    dm_violated_qubits: StringVar
    dm_violated_qubits_error_msg: StringVar

    # plot primary mesh
    pm_show_labels: BooleanVar
    pm_violated_qubits: StringVar
    pm_violated_qubits_error_msg: StringVar
    pm_transparent_faces: BooleanVar

    plotter: Plotter = None
    dual_graph_mesh: pyvista.PolyData = None

    def __init__(self, root: Tk, pool: Executor, code_config: CodeConfig) -> None:
        self.root = root
        self.pool = pool
        self.code_config = code_config

        self.use_edge_color = BooleanVar(value=False)
        self.edges_between_boundaries = BooleanVar(value=True)
        self.exclude_boundaries = BooleanVar(value=False)

        self.dm_violated_qubits = StringVar()
        self.dm_violated_qubits_error_msg = StringVar()

        self.pm_show_labels = BooleanVar(value=False)
        self.pm_violated_qubits = StringVar()
        self.pm_violated_qubits_error_msg = StringVar()
        self.pm_transparent_faces = BooleanVar(value=True)

        self.root.bind("<<DualGraphCreationFinished>>", self._update_plotter, add="+")

    @property
    def plotter_class(self) -> Optional[type[Plotter]]:
        if self.code_config.codetype:
            return self.code_config.codetype.plotter_class
        return None

    def _update_plotter(self, _) -> None:
        cfg = self.code_config
        self.plotter = self.plotter_class(dual_graph=cfg.dual_graph, distance=cfg.distance)

    @staticmethod
    def _create_state_change_callback(*args, state: str) -> Callable:
        def callback(*_) -> None:
            change_state(args, state=state)
        return callback

    def _create_violatedqubits_validator(self, value_store: StringVar, error_store: StringVar, submits: list[ttk.Button]) -> Callable:
        def validator(new_value: str, operation: str) -> bool:
            error_store.set('')
            if new_value == "":
                return True
            raw_qubits = new_value.split(",")
            qubits = []
            for q in raw_qubits:
                qubit = q.strip()
                if qubit == "":
                    continue
                if not qubit.isdigit():
                    error_store.set("Qubits are numbers.")
                    return False
                qubits.append(qubit)

            if operation in {'focusout', 'focusin', 'forced'}:
                value_store.set(",".join(qubits))

            return True

        return validator

    def _highlighted_nodes(self, error_qubits: list[int]) -> list[DualGraphNode]:
        return [node for node in self.code_config.dual_graph.nodes()
                if len(set(node.qubits) & set(error_qubits)) % 2 and not node.is_boundary]

    def _create_dualmesh_create_command(self, all_ttk: list[ttk.Widget], progressbar: ttk.Progressbar) -> Callable:
        def command() -> None:
            def callback(f: Future) -> None:
                self.dual_graph_mesh = f.result()
                progressbar.stop()
                change_state(all_ttk, state="normal")
                self.root.event_generate("<<DualMeshCreationFinished>>")

            progressbar.start()
            change_state(all_ttk, state="disabled")
            self.root.event_generate("<<DualMeshCreationStarted>>")
            # highlight all nodes to make them more distinguishable
            highlighted_nodes = self.code_config.dual_graph.nodes()
            if (violated_qubits := self.dm_violated_qubits.get()):
                qubits = sorted(set([int(qubit) for qubit in violated_qubits.split(",")]))
                highlighted_nodes = self._highlighted_nodes(qubits)
            highlighted_edges = None
            # to ensure better visibility
            if self.use_edge_color.get():
                highlighted_edges = self.code_config.dual_graph.edges()
            f_: Future = self.pool.submit(
                self.plotter.construct_dual_mesh, self.code_config.dual_graph,
                use_edges_colors=self.use_edge_color.get(),
                highlighted_nodes=highlighted_nodes,
                highlighted_edges=highlighted_edges,
                include_edges_between_boundaries=self.edges_between_boundaries.get(),
                exclude_boundaries=self.exclude_boundaries.get(),
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
        violated_qubits = ttk.Entry(frame)
        submit = ttk.Button(frame)
        all_ttk = [use_edge_color, edges_between_boundaries, exclude_boundaries, violated_qubits, submit]
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

        violated_qubits_label = ttk.Label(frame, text="Errors on qubits")
        validate_qubits_wrapper = (
            frame.register(self._create_violatedqubits_validator(self.dm_violated_qubits, self.dm_violated_qubits_error_msg, submit)),
            '%P', '%V',
        )
        violated_qubits.configure(textvariable=self.dm_violated_qubits, validate='all', validatecommand=validate_qubits_wrapper)
        violated_qubits_msg = ttk.Label(frame, font='TkSmallCaptionFont', foreground='red', textvariable=self.dm_violated_qubits_error_msg)
        violated_qubits_label.grid(row=20, column=0)
        violated_qubits.grid(row=20, column=1, columnspan=2, sticky="ew")
        violated_qubits_msg.grid(row=21, column=1, columnspan=2, padx=5, pady=5, sticky='w')

        submit.configure(text="Build", command=self._create_dualmesh_create_command(all_ttk, progress_bar))
        submit.grid(row=100, column=1, sticky="se")
        progress_bar.configure(orient="horizontal", mode="indeterminate")
        progress_bar.grid(row=100, column=0, sticky="sw")
        frame.rowconfigure(100, weight=1)

        return frame

    def _dualmesh_plot_command(self) -> None:
        self.pool.submit(
            self.plotter.plot_debug_mesh, self.dual_graph_mesh,
            window_title=self.code_config.code_description,
        )

    def create_dual_plot_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, borderwidth=5, relief="ridge", padding=(3, 3, 12, 12), text="Plot Dual Graph")
        frame.configure()
        frame.columnconfigure(1, weight=1)

        submit_button = ttk.Button(frame)
        all_ttk = [submit_button]

        self.root.bind("<<DualGraphCreationStarted>>", self._create_state_change_callback(*all_ttk, state="disabled"), add="+")
        self.root.bind("<<DualMeshCreationStarted>>", self._create_state_change_callback(*all_ttk, state="disabled"), add="+")
        self.root.bind("<<DualMeshCreationFinished>>", self._create_state_change_callback(*all_ttk, state="normal"), add="+")
        for element in all_ttk:
            element.configure(state="disabled")

        submit_button.configure(text="Plot", command=self._dualmesh_plot_command)
        submit_button.grid(row=10, column=1, sticky="se")
        frame.rowconfigure(10, weight=1)

        return frame

    def _create_primarymesh_plot_command(self, include_dual_mesh: bool) -> Callable:
        def command() -> None:
            func = self.plotter.plot_primary_mesh
            args = []
            if include_dual_mesh:
                func = self.plotter.plot_debug_primary_mesh
                args = [self.dual_graph_mesh]
            violated_qubits = None
            highlighted_volumes = None
            if (qubits := self.pm_violated_qubits.get()):
                violated_qubits = sorted(set([int(qubit) for qubit in qubits.split(",")]))
                highlighted_volumes = self._highlighted_nodes(violated_qubits)
            self.pool.submit(
                func, *args,
                show_qubit_labels=self.pm_show_labels.get(),
                highlighted_qubits=violated_qubits,
                highlighted_volumes=highlighted_volumes,
                transparent_faces=self.pm_transparent_faces.get(),
                color_edges=True,
                window_title=self.code_config.code_description,
            )
        return command

    def create_primary_plot_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, borderwidth=5, relief="ridge", padding=(3, 3, 12, 12), text="Plot Primary Graph")
        frame.configure()
        frame.columnconfigure(1, weight=1)

        show_labels = ttk.Checkbutton(frame)
        transparent_faces = ttk.Checkbutton(frame)
        violated_qubits = ttk.Entry(frame)
        submit_button = ttk.Button(frame)
        submit_with_dual_mesh = ttk.Button(frame)
        submit_ttks = [show_labels, violated_qubits, transparent_faces, violated_qubits, submit_button]
        all_ttks = [*submit_ttks, submit_with_dual_mesh]

        self.root.bind("<<DualGraphCreationStarted>>", self._create_state_change_callback(*all_ttks, state="disabled"), add="+")
        self.root.bind("<<DualGraphCreationFinished>>", self._create_state_change_callback(*submit_ttks, state="normal"), add="+")
        # additionaly bind submit_with_dual_mesh to respective events
        self.root.bind("<<DualMeshCreationStarted>>", self._create_state_change_callback(submit_with_dual_mesh, state="disabled"), add="+")
        self.root.bind("<<DualMeshCreationFinished>>", self._create_state_change_callback(submit_with_dual_mesh, state="normal"), add="+")
        for element in all_ttks:
            element.configure(state="disabled")

        show_labels.configure(variable=self.pm_show_labels, text="Show qubit labels")
        show_labels.grid(row=0, column=1, sticky="w")

        transparent_faces.configure(variable=self.pm_transparent_faces, text="Render faces transparent")
        transparent_faces.grid(row=10, column=1, sticky="w")

        violated_qubits_label = ttk.Label(frame, text="Errors on qubits")
        validate_qubits_wrapper = (frame.register(self._create_violatedqubits_validator(
            self.pm_violated_qubits, self.pm_violated_qubits_error_msg, [submit_button, submit_with_dual_mesh])), '%P', '%V')
        violated_qubits.configure(textvariable=self.pm_violated_qubits, validate='all', validatecommand=validate_qubits_wrapper)
        violated_qubits_msg = ttk.Label(frame, font='TkSmallCaptionFont', foreground='red', textvariable=self.pm_violated_qubits_error_msg)
        violated_qubits_label.grid(row=20, column=0)
        violated_qubits.grid(row=20, column=1, columnspan=2, sticky="ew")
        violated_qubits_msg.grid(row=21, column=1, columnspan=2, padx=5, pady=5, sticky='w')

        submit_with_dual_mesh.configure(text="Plot (with Dual)", command=self._create_primarymesh_plot_command(include_dual_mesh=True))
        submit_with_dual_mesh.grid(row=100, column=0, sticky="se")
        submit_button.configure(text="Plot", command=self._create_primarymesh_plot_command(include_dual_mesh=False))
        submit_button.grid(row=100, column=1, sticky="se")
        frame.rowconfigure(100, weight=1)

        return frame


class QCodePlot3dGUI:
    root: Tk
    pool: Executor
    code_config: CodeConfig

    def __init__(self) -> None:
        self.root = Tk()
        self.pool = ProcessPoolExecutor()
        self.root.title("qCodePlot3D")

        content = ttk.Frame(self.root, padding=(3, 3, 12, 12))
        content.grid(row=0, column=0, sticky="nsew")

        self.code_config = CodeConfig(self.root, self.pool)
        code_config_frame = self.code_config.create_frame(content)
        code_config_frame.grid(row=0, column=0, sticky="nsew")

        self.plotter_config = PlotterConfig(self.root, self.pool, self.code_config)
        plotter_config_frame = self.plotter_config.create_dual_config_frame(content)
        plotter_config_frame.grid(row=0, column=10, sticky="nsew")
        plotter_dm_plot_frame = self.plotter_config.create_dual_plot_frame(content)
        plotter_dm_plot_frame.grid(row=10, column=0, sticky="nsew")
        plotter_pm_plot_frame = self.plotter_config.create_primary_plot_frame(content)
        plotter_pm_plot_frame.grid(row=10, column=10, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(10, weight=1)

        def on_closing():
            self.pool.shutdown(wait=False, cancel_futures=True)
            try:
                parent = psutil.Process(os.getpid())
            except psutil.NoSuchProcess:
                pass
            else:
                children = parent.children(recursive=True)
                for process in children:
                    try:
                        process.send_signal(signal.SIGTERM)
                    except psutil.NoSuchProcess:
                        pass
            self.root.destroy()

        # add graceful shutdown
        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        # enter event loop
        self.root.mainloop()
