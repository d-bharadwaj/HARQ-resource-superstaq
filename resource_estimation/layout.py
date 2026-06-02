# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import cirq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from . import lattice_surgery_primitives as lsp
from collections import deque
from dataclasses import dataclass
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from typing import Literal
from math import ceil, sqrt
from itertools import combinations, product

_EMPTY = 0
_DATA = 1
_ANCILLA = 2
_S_FACTORY = 3
_T_FACTORY = 4
_MEMORY = 5
_COMPUTE = 6
_COMMUNICATION = 7

_LAYOUT_CMAP = ListedColormap(
    [
        "white",  # empty
        "#A8E6A1",  # data qubit — light green
        "#F5BFC8",  # ancilla patch — light pink
        "#FFD4A3",  # S factory — light orange
        "#6EC6E6",  # T factory — cyan
        "#B9D7F5",  # memory region — light blue
        "#B8E3C2",  # compute region — green
        "#D8C7F2",  # communication region — lavender
    ]
)
_LAYOUT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], _LAYOUT_CMAP.N)
_LAYOUT_LABELS = {
    _DATA: "Data qubit",
    _ANCILLA: "Ancilla qubit",
    _S_FACTORY: "S factory",
    _T_FACTORY: "T factory",
    _MEMORY: "Memory region",
    _COMPUTE: "Compute region",
    _COMMUNICATION: "Communication region",
}


@dataclass
class Layout(abc.ABC):
    """
    Base class for layouts used by the fault tolerant compiler to track factory use and CNOT routing
    """

    input_circuit: cirq.Circuit
    num_t_factories: int = 0
    num_s_factories: int = 0

    def __post_init__(self):
        self.mapped_circuit = None
        self.layout_graph = None
        self._available_t_factories = deque()
        self._available_s_factories = deque()
        self._all_factories = set()
        self._generate()

    def set_map_circuit(self, qubit_map: dict[cirq.Qid, cirq.GridQubit]) -> None:
        """
        Apply a given mapping from qubits in the input circuit to GridQubits used for compilation
        """
        mapped_circuit = cirq.Circuit(
            moment.transform_qubits(qubit_map) for moment in self.input_circuit
        )
        self.mapped_circuit = mapped_circuit

    def reset_graph(self) -> None:
        """
        Reset the graph to its starting state by setting all factory qubits to the `used` state
        """
        G = self.layout_graph
        for node in G.nodes:
            if G.nodes[node]["patch_type"] == "factory":
                G.nodes[node]["used"] = True
        # Resets the available factories
        self._available_t_factories = deque()
        self._available_s_factories = deque()

    def reload_factories(self, ftype: Literal["t", "s"]) -> None:
        if ftype == "t":
            all_t_factories = [
                factory
                for factory in self._all_factories
                if self.layout_graph.nodes[factory]["ftype"] == "t"
            ]
            self._available_t_factories = deque(all_t_factories)
        elif ftype == "s":
            all_s_factories = [
                factory
                for factory in self._all_factories
                if self.layout_graph.nodes[factory]["ftype"] == "s"
            ]
            self._available_s_factories = deque(all_s_factories)
        else:
            raise ValueError(f"{ftype} is not a valid factory type")
        # Update graph to reflect the new status
        for node in self.layout_graph.nodes:
            if node in self.available_s_factories or node in self._available_t_factories:
                self.layout_graph.nodes[node]["used"] = False

    def _generate(self) -> None:
        """
        Private method to generate the underlying networkx graph, qubit map, and qubit placement
        This method is the core of what defines a Layout
        At this level, the graph generated has no locality, but methods in subclasses should be local (especially lattice surgery layouts)
        """
        total_qubits = (
            len(self.input_circuit.all_qubits()) + self.num_s_factories + self.num_t_factories
        )
        side_length = ceil(sqrt(total_qubits))

        def idx_to_xy(idx: int) -> tuple[int, int]:
            x = idx // side_length
            y = idx % side_length
            return x, y

        qubit_map = {
            qid: cirq.GridQubit(*idx_to_xy(idx))
            for idx, qid in enumerate(sorted(self.input_circuit.all_qubits()))
        }
        self.set_map_circuit(qubit_map=qubit_map)
        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [
                (
                    cirq.GridQubit(*idx_to_xy(idx + len(G.nodes))),
                    dict(patch_type="factory", ftype="t", used=True),
                )
                for idx in range(self.num_t_factories)
            ],
        )
        G.add_nodes_from(
            [
                (
                    cirq.GridQubit(*idx_to_xy(idx + len(G.nodes))),
                    dict(patch_type="factory", ftype="s", used=True),
                )
                for idx in range(self.num_s_factories)
            ],
        )
        G.add_edges_from((n1, n2) for n1, n2 in combinations(G.nodes, 2))
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G

    @property
    def available_t_factories(self) -> deque[cirq.GridQubit]:
        return self._available_t_factories

    @property
    def available_s_factories(self) -> deque[cirq.GridQubit]:
        return self._available_s_factories

    def nearest_factory(self, qubit: cirq.GridQubit, ftype: Literal["s", "t"]) -> cirq.GridQubit:
        """
        Finds the closest factory of desired type according to the Manhattan distance using the GridQubit indices of the factory qubits that do not have the `used` status
        Removes the returned factory from the available options and sets its status to `used`
        """
        available_factories = (
            self.available_s_factories if ftype == "s" else self.available_t_factories
        )
        if not available_factories:
            raise ValueError(f"No available {ftype} factories available!")
        r, c = qubit.row, qubit.col
        # Closest factory according to the L1 distance
        factory = min(available_factories, key=lambda fact: abs(fact.row - r) + abs(fact.col - c))
        # Factory now used must be removed
        self.layout_graph.nodes[factory]["used"] = True
        available_factories.remove(factory)
        if ftype == "s":
            self._available_s_factories = available_factories
        else:
            self._available_t_factories = available_factories
        return factory

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit) -> list[cirq.GridQubit]:
        """
        Finds the patches required to perform a lattice surgery CNOT between two logical qubits
        The path returned must include at least one ancilla
        This method does not account for other CNOTs in the logical circuit, so choosing the shortest path might not correspond to the optimal path
        """
        # TODO: See if there is a way to maximize parallelism, or port over work that already does this maximization
        G = self.layout_graph

        def custom_weight(u: cirq.GridQubit, v: cirq.GridQubit, attr: dict) -> int | None:
            if (G.nodes[v]["patch_type"] == "data") or (G.nodes[v]["patch_type"] == "factory"):
                # Must go through at least one ancilla
                if (v == trgt and u == ctrl) or (u == trgt and v == ctrl):
                    return None
                elif v == trgt or v == ctrl:
                    return 1
                else:
                    return None
            return 1

        path = nx.dijkstra_path(G=G, source=ctrl, target=trgt, weight=custom_weight)
        return path

    def draw(self, show_legend: bool = True) -> None:  # pragma: no cover
        """
        Draw method to display layouts clearly
        """
        color_dict = {
            _T_FACTORY: _LAYOUT_CMAP.colors[_T_FACTORY],
            _S_FACTORY: _LAYOUT_CMAP.colors[_S_FACTORY],
            _DATA: _LAYOUT_CMAP.colors[_DATA],
            _ANCILLA: _LAYOUT_CMAP.colors[_ANCILLA],
            _MEMORY: _LAYOUT_CMAP.colors[_MEMORY],
            _COMPUTE: _LAYOUT_CMAP.colors[_COMPUTE],
            _COMMUNICATION: _LAYOUT_CMAP.colors[_COMMUNICATION],
        }
        G = self.layout_graph
        node_color = []
        for node in G.nodes:
            node_color.append(color_dict[self._patch_code(G.nodes[node])])
        pos = {node: (node.row, node.col) for node in G.nodes}
        nx.draw(G, with_labels=True, node_color=node_color, pos=pos)
        if show_legend:
            plt.gca().legend(handles=self._legend_handles(), loc="upper left", bbox_to_anchor=(1, 1))

    def _patch_code(self, node_dict: dict) -> int:
        region = node_dict.get("region")
        if region == "memory":
            return _MEMORY
        if region == "compute":
            return _COMPUTE
        if region == "communication":
            return _COMMUNICATION
        if node_dict["patch_type"] == "data":
            return _DATA
        if node_dict["patch_type"] == "ancilla":
            return _ANCILLA
        if node_dict["ftype"] == "s":
            return _S_FACTORY
        return _T_FACTORY

    def _layout_grid(self, grid_size: int | None = None, padding: int = 2) -> np.ndarray:
        G = self.layout_graph
        if not G.nodes:
            return np.zeros((1, 1), dtype=int)

        rows = [node.row for node in G.nodes]
        cols = [node.col for node in G.nodes]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        if grid_size is not None:
            height = width = grid_size
            pad_top = (grid_size - (max_row - min_row + 1)) // 2
            pad_left = (grid_size - (max_col - min_col + 1)) // 2
            row_offset = pad_top - min_row
            col_offset = pad_left - min_col
        else:
            min_row -= padding
            min_col -= padding
            max_row += padding
            max_col += padding
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            row_offset = -min_row
            col_offset = -min_col

        grid = np.zeros((height, width), dtype=int)
        for node in G.nodes:
            grid[node.row + row_offset, node.col + col_offset] = self._patch_code(G.nodes[node])
        return grid

    def _legend_handles(self) -> list[Patch]:
        present_codes = sorted({self._patch_code(self.layout_graph.nodes[node]) for node in self.layout_graph})
        return [
            Patch(facecolor=_LAYOUT_CMAP.colors[code], edgecolor="lightgray", label=_LAYOUT_LABELS[code])
            for code in present_codes
            if code in _LAYOUT_LABELS
        ]

    def visualize_layout(
        self,
        title: str | None = None,
        grid_size: int | None = None,
        padding: int = 2,
        ax: plt.Axes | None = None,
        show: bool = True,
        show_legend: bool = True,
    ) -> plt.Axes | None:  # pragma: no cover
        """
        Render the layout as a colored grid.

        Colors: white (empty), light green (data qubits), light pink (ancilla),
        light orange (S factories), cyan (T factories). Region-aware layouts
        additionally show memory in blue, compute in green, and communication in purple.
        """
        grid = self._layout_grid(grid_size=grid_size, padding=padding)
        height, width = grid.shape

        created_fig = ax is None
        if created_fig:
            cell_size = 0.35
            fig, ax = plt.subplots(figsize=(width * cell_size, height * cell_size), dpi=100)

        x_edges = np.arange(width + 1) - 0.5
        y_edges = np.arange(height + 1) - 0.5
        ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            cmap=_LAYOUT_CMAP,
            norm=_LAYOUT_NORM,
            shading="flat",
            edgecolors="lightgray",
            linewidth=0.5,
            antialiased=False,
        )
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color("black")
        if title is not None:
            ax.set_title(title)
        if show_legend:
            ax.legend(handles=self._legend_handles(), loc="upper left", bbox_to_anchor=(1, 1))
        if show and created_fig:
            plt.show()
            return None
        return ax


class MovementLayout(Layout):
    """
    Layout class representing the connections available to Movement Architectures
    It does not have S factories and the number of T factories is fully configurable
    The current implementation assumes all-to-all connectivity in the logical qubit layout because the cost for nonlocal moves is handled deeper in the stack
    A better implementation might do a smart placement of qubits on the grid to minimize overall distance travelled
    """

    # TODO: build this implementation
    def __init__(self, input_circuit: cirq.Circuit, num_t_factories: int = 1):
        super().__init__(
            input_circuit=input_circuit, num_t_factories=num_t_factories, num_s_factories=0
        )

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit):
        raise NotImplementedError


class Heterogenous_MovementLayout(Layout):
    """
    Movement layout with explicit memory, compute, communication, and factory regions.

    Logical circuit qubits start in memory. Each input operation is serialized by moving its
    operands into compute slots, applying the operation there, and moving the operands back.
    """

    def __init__(
        self,
        input_circuit: cirq.Circuit,
        num_t_factories: int = 1,
        num_s_factories: int | None = None,
        compute_capacity: int = 4,
        communication_width: int | None = None,
    ):
        self.compute_capacity = compute_capacity
        self.communication_width = communication_width
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_s_factories=num_t_factories if num_s_factories is None else num_s_factories,
        )

    def _generate(self) -> None:
        all_qubits = sorted(self.input_circuit.all_qubits())
        if self.compute_capacity < 1:
            raise ValueError("compute_capacity must be at least 1")
        communication_width = (
            max(2, ceil(len(all_qubits) / 10))
            if self.communication_width is None
            else self.communication_width
        )
        if communication_width < 1:
            raise ValueError("communication_width must be at least 1")
        max_arity = max((len(op.qubits) for op in self.input_circuit.all_operations()), default=1)
        if self.compute_capacity < max_arity:
            raise ValueError(
                "compute_capacity must be at least as large as the widest operation "
                f"in the circuit ({max_arity})"
            )

        memory_width = max(1, ceil(sqrt(len(all_qubits))))
        memory_height = max(1, ceil(len(all_qubits) / memory_width))
        compute_width = min(2, self.compute_capacity)
        compute_height = ceil(self.compute_capacity / compute_width)
        factory_count = self.num_t_factories + self.num_s_factories
        height = max(memory_height, compute_height + factory_count, 1)

        memory_positions = [
            cirq.GridQubit(row, col)
            for row in range(memory_height)
            for col in range(memory_width)
        ][: len(all_qubits)]
        communication_start_col = memory_width
        communication = [
            cirq.GridQubit(row, communication_start_col + col)
            for row in range(height)
            for col in range(communication_width)
        ]

        compute_col = communication_start_col + communication_width
        compute_positions = [
            cirq.GridQubit(row, compute_col + col)
            for row in range(compute_height)
            for col in range(compute_width)
        ][: self.compute_capacity]

        factory_col = compute_col
        factory_start_row = compute_height
        t_factories = [
            cirq.GridQubit(factory_start_row + row, factory_col)
            for row in range(self.num_t_factories)
        ]
        s_factories = [
            cirq.GridQubit(factory_start_row + row + self.num_t_factories, factory_col)
            for row in range(self.num_s_factories)
        ]

        qubit_map = dict(zip(all_qubits, memory_positions))
        self._memory_map = qubit_map
        self._compute_slots = compute_positions

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data", region="memory")) for q in memory_positions],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="data", region="compute")) for q in compute_positions],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla", region="communication")) for q in communication],
        )
        G.add_nodes_from(
            [
                (q, dict(patch_type="factory", ftype="t", region="factory", used=True))
                for q in t_factories
            ],
        )
        G.add_nodes_from(
            [
                (q, dict(patch_type="factory", ftype="s", region="factory", used=True))
                for q in s_factories
            ],
        )
        for node in G.nodes:
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = cirq.GridQubit(node.row + d_row, node.col + d_col)
                if neighbor in G:
                    G.add_edge(node, neighbor)
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G
        self.mapped_circuit = self._memory_compute_circuit(qubit_map, compute_positions)
        # Make passive regional capacity visible to circuit-based physical-qubit accounting.
        self.mapped_circuit = cirq.Circuit(cirq.I.on_each(*sorted(G.nodes))) + self.mapped_circuit

    def _memory_compute_circuit(
        self,
        qubit_map: dict[cirq.Qid, cirq.GridQubit],
        compute_slots: list[cirq.GridQubit],
    ) -> cirq.Circuit:
        regional_circuit = cirq.Circuit()
        for op in self.input_circuit.all_operations():
            op_qubits = list(op.qubits)
            active_slots = compute_slots[: len(op_qubits)]
            to_compute = {
                qubit: slot for qubit, slot in zip(op_qubits, active_slots)
            }
            for qubit, slot in to_compute.items():
                regional_circuit += self._moves_along_path(
                    self.route_through_communication(qubit_map[qubit], slot)
                )
            regional_circuit += cirq.Moment(op.transform_qubits(to_compute))
            for qubit, slot in to_compute.items():
                regional_circuit += self._moves_along_path(
                    self.route_through_communication(slot, qubit_map[qubit])
                )
        return regional_circuit

    def route_through_communication(
        self,
        source: cirq.GridQubit,
        target: cirq.GridQubit,
    ) -> list[cirq.GridQubit]:
        """
        Find a shortest route between regions while forcing intermediate hops through communication.
        """
        graph = self.layout_graph

        def weight(
            left: cirq.GridQubit,
            right: cirq.GridQubit,
            _attrs: dict,
        ) -> int | None:
            left_region = graph.nodes[left].get("region")
            right_region = graph.nodes[right].get("region")
            if left_region == right_region:
                return 1
            if "communication" in (left_region, right_region):
                return 1
            return None

        return nx.dijkstra_path(graph, source=source, target=target, weight=weight)

    def route_factory_to_compute(
        self,
        factory: cirq.GridQubit,
        compute_qubit: cirq.GridQubit,
    ) -> list[cirq.GridQubit]:
        """
        Find the communication-mediated path needed for factory/data teleportation.
        """
        return self.route_through_communication(factory, compute_qubit)

    def nearest_factory(self, qubit: cirq.GridQubit, ftype: Literal["s", "t"]) -> cirq.GridQubit:
        """
        Select the available factory with the shortest communication-mediated route.
        """
        available_factories = (
            self.available_s_factories if ftype == "s" else self.available_t_factories
        )
        if not available_factories:
            raise ValueError(f"No available {ftype} factories available!")

        factory = min(
            available_factories,
            key=lambda candidate: len(self.route_factory_to_compute(candidate, qubit)),
        )
        self.layout_graph.nodes[factory]["used"] = True
        available_factories.remove(factory)
        if ftype == "s":
            self._available_s_factories = available_factories
        else:
            self._available_t_factories = available_factories
        return factory

    def moves_for_factory_to_compute(
        self,
        factory: cirq.GridQubit,
        compute_qubit: cirq.GridQubit,
    ) -> list[cirq.Operation]:
        """
        Movement operations representing the route that makes a factory/data CNOT local.
        """
        return list(
            self._moves_along_path(
                self.route_factory_to_compute(factory, compute_qubit)
            ).all_operations()
        )

    def _moves_along_path(self, path: list[cirq.GridQubit]) -> cirq.Circuit:
        operations = []
        for left, right in zip(path, path[1:]):
            regions = {
                self.layout_graph.nodes[left].get("region"),
                self.layout_graph.nodes[right].get("region"),
            }
            if "communication" in regions:
                move_gate = lsp.CommunicationMove(route_distance=1)
            else:
                move_gate = lsp.Move(zone=None, route_distance=1)
            operations.append(move_gate.on(left, right))
        return cirq.Circuit(operations)

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit):
        raise NotImplementedError


class Square(Heterogenous_MovementLayout):
    """
    Regional layout with SSM memory/bus on the left and SSOQ compute/factories on the right.

    The communication bus is a fixed-width vertical strip between memory and the
    right-side regions. Logical qubits begin in memory and are staged into the
    compute quadrant before operations are compiled.
    """

    def __init__(
        self,
        input_circuit: cirq.Circuit,
        num_t_factories: int = 1,
        num_s_factories: int | None = None,
        compute_capacity: int = 4,
        communication_width: int = 2,
    ):
        super().__init__(
            input_circuit=input_circuit,
            num_t_factories=num_t_factories,
            num_s_factories=num_t_factories if num_s_factories is None else num_s_factories,
            compute_capacity=compute_capacity,
            communication_width=communication_width,
        )

    def _generate(self) -> None:
        all_qubits = sorted(self.input_circuit.all_qubits())
        if self.compute_capacity < 1:
            raise ValueError("compute_capacity must be at least 1")
        if self.communication_width != 2:
            raise ValueError("Square uses a fixed 2-column communication bus")
        max_arity = max((len(op.qubits) for op in self.input_circuit.all_operations()), default=1)
        if self.compute_capacity < max_arity:
            raise ValueError(
                "compute_capacity must be at least as large as the widest operation "
                f"in the circuit ({max_arity})"
            )

        slots_per_row = max(1, ceil(sqrt(self.compute_capacity)))
        compute_rows = ceil(self.compute_capacity / slots_per_row)
        compute_height = max(3, 2 * compute_rows - 1)
        right_width = max(3, 2 * slots_per_row - 1)
        memory_width = max(2, right_width)
        factory_count = self.num_t_factories + self.num_s_factories
        factory_slots_per_row = max(1, (right_width + 1) // 2)
        factory_rows = ceil(max(1, factory_count) / factory_slots_per_row)
        factory_height = max(3, 2 * factory_rows - 1)
        height = max(
            2 * compute_height,
            compute_height + factory_height,
            ceil(max(1, len(all_qubits)) / memory_width),
        )

        communication_start_col = memory_width
        right_start_col = communication_start_col + self.communication_width
        memory_cells = [
            cirq.GridQubit(row, col)
            for row in range(height)
            for col in range(memory_width)
        ]
        communication_cells = [
            cirq.GridQubit(row, col)
            for row in range(height)
            for col in range(communication_start_col, right_start_col)
        ]
        compute_cells = [
            cirq.GridQubit(row, col)
            for row in range(compute_height)
            for col in range(right_start_col, right_start_col + right_width)
        ]
        factory_cells = [
            cirq.GridQubit(row, col)
            for row in range(compute_height, height)
            for col in range(right_start_col, right_start_col + right_width)
        ]
        compute_positions = [
            cirq.GridQubit(row, col)
            for row in range(0, compute_height, 2)
            for col in range(right_start_col, right_start_col + right_width, 2)
        ][: self.compute_capacity]
        factory_slot_positions = [
            cirq.GridQubit(row, col)
            for row in range(compute_height, height, 2)
            for col in range(right_start_col, right_start_col + right_width, 2)
        ][:factory_count]
        t_factories = factory_slot_positions[: self.num_t_factories]
        s_factories = factory_slot_positions[
            self.num_t_factories : self.num_t_factories + self.num_s_factories
        ]
        factory_positions = set(t_factories + s_factories)
        qubit_map = dict(zip(all_qubits, memory_cells[: len(all_qubits)]))

        self._memory_map = qubit_map
        self._compute_slots = compute_positions

        G = nx.Graph()
        G.add_nodes_from(
            [
                (
                    q,
                    dict(
                        patch_type="data" if q in qubit_map.values() else "ancilla",
                        region="memory",
                    ),
                )
                for q in memory_cells
            ],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla", region="communication")) for q in communication_cells],
        )
        G.add_nodes_from(
            [
                (
                    q,
                    dict(
                        patch_type="data" if q in compute_positions else "ancilla",
                        region="compute",
                    ),
                )
                for q in compute_cells
            ],
        )
        G.add_nodes_from(
            [
                (
                    q,
                    dict(
                        patch_type="factory" if q in factory_positions else "ancilla",
                        ftype="t" if q in t_factories else "s" if q in s_factories else None,
                        region="factory",
                        used=True if q in factory_positions else None,
                    ),
                )
                for q in factory_cells
            ],
        )
        for node in G.nodes:
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = cirq.GridQubit(node.row + d_row, node.col + d_col)
                if neighbor in G:
                    G.add_edge(node, neighbor)

        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G
        self.mapped_circuit = self._memory_compute_circuit(qubit_map, compute_positions)
        self.mapped_circuit = cirq.Circuit(cirq.I.on_each(*sorted(G.nodes))) + self.mapped_circuit

    def route_factory_to_compute(
        self,
        factory: cirq.GridQubit,
        compute_qubit: cirq.GridQubit,
    ) -> list[cirq.GridQubit]:
        return nx.shortest_path(self.layout_graph, source=factory, target=compute_qubit)

    def moves_for_factory_to_compute(
        self,
        factory: cirq.GridQubit,
        compute_qubit: cirq.GridQubit,
    ) -> list[cirq.Operation]:
        return []

    def _moves_along_path(self, path: list[cirq.GridQubit]) -> cirq.Circuit:
        operations = []
        for left, right in zip(path, path[1:]):
            left_region = self.layout_graph.nodes[left].get("region")
            right_region = self.layout_graph.nodes[right].get("region")
            regions = {left_region, right_region}
            if left_region == right_region:
                if left_region == "memory":
                    operations.append(lsp.Move(zone=None, route_distance=1).on(left, right))
                elif left_region == "communication":
                    operations.append(lsp.CommunicationMove(route_distance=1).on(left, right))
                continue
            if regions <= {"memory", "communication"}:
                operations.append(lsp.CommunicationMove(route_distance=1).on(left, right))
            elif "communication" in regions and ("compute" in regions or "factory" in regions):
                operations.append(
                    lsp.ModalityTransfer(source=left_region, target=right_region).on(left, right)
                )
            elif regions <= {"compute", "factory"}:
                continue
            else:
                operations.append(
                    lsp.ModalityTransfer(source=left_region, target=right_region).on(left, right)
                )
        return cirq.Circuit(operations)

    def route_cnot(self, ctrl: cirq.GridQubit, trgt: cirq.GridQubit):
        return Layout.route_cnot(self, ctrl=ctrl, trgt=trgt)


class Column(Layout):
    """
    Lattice surgery Layout based on having two columns of logical qubits
    S | a | q | a | q | a | S
    T | a | a | a | a | a | T
    S | a | q | a | q | a | S
    T | a | a | a | a | a | T
    ...
    """

    def __init__(self, input_circuit: cirq.Circuit):
        rows = ceil(len(input_circuit.all_qubits()) / 2)
        num_s_factories = 2 * rows
        num_t_factories = 2 * rows
        super().__init__(
            input_circuit=input_circuit,
            num_s_factories=num_s_factories,
            num_t_factories=num_t_factories,
        )

    def _generate(self) -> None:
        """
        Places and assigns logical qubits according to the column configuration
        In the case where the number of logical qubits is odd fill the would-be logical qubit with an ancilla
        """
        qubit_map: dict[cirq.Qid, cirq.GridQubit] = {}
        all_qubits = list(self.input_circuit.all_qubits())
        s_factories = []
        t_factories = []
        ancillas = []
        num_rows = ceil(len(all_qubits) / 2)
        for idx, qid in enumerate(sorted(all_qubits)):
            row = 2 * (idx // 2)
            col = 4 if idx % 2 else 2
            qubit_map[qid] = cirq.GridQubit(row, col)
        self.set_map_circuit(qubit_map=qubit_map)
        for row in range(2 * num_rows):
            if row % 2 == 0:
                s_factories.extend([cirq.GridQubit(row, 0), cirq.GridQubit(row, 6)])
                ancillas.extend(
                    [cirq.GridQubit(row, 1), cirq.GridQubit(row, 3), cirq.GridQubit(row, 5)]
                )
            else:
                t_factories.extend([cirq.GridQubit(row, 0), cirq.GridQubit(row, 6)])
                ancillas.extend([cirq.GridQubit(row, col) for col in range(1, 6)])
        if len(all_qubits) % 2:
            ancillas.append(cirq.GridQubit(2 * num_rows - 2, 4))

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="t", used=True)) for q in t_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="s", used=True)) for q in s_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        # Connect nearest neighbors (Manhattan distance 1) without O(n^2) pairwise checks
        for node in G.nodes:
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = cirq.GridQubit(node.row + d_row, node.col + d_col)
                if neighbor in G:
                    G.add_edge(node, neighbor)
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G


class FactorySandwich(Layout):
    """
    Lattice surgery layout based on having a line of logical qubits sandwiched by factory qubits and ancilla
    S | S | ... | S
    a | a | ... | a
    q | q | ... | q
    a | a | ... | a
    T | T | ... | T

    Because the numbers of S and T factories are configurable, the dimensions might not line up resulting in things like
    S | S | S
    a | a | a | a | a
    q | q | q | q | q
    a | a | a | a | a
    T | T | T | T
    """

    def _generate(self):
        """
        Places and assigns logical qubits according to the Sandwich configuration
        """
        qubit_map: dict[cirq.Qid, cirq.GridQubit] = {}
        all_qubits = list(self.input_circuit.all_qubits())
        length = max(len(all_qubits), self.num_t_factories, self.num_s_factories)
        s_factories = []
        t_factories = []
        ancillas = []
        for idx, qid in enumerate(sorted(all_qubits)):
            qubit_map[qid] = cirq.GridQubit(2, idx)
        self.set_map_circuit(qubit_map=qubit_map)
        ancillas = [cirq.GridQubit(row, idx) for idx in range(length) for row in (1, 3)]
        s_factories = [cirq.GridQubit(0, idx) for idx in range(self.num_s_factories)]
        t_factories = [cirq.GridQubit(4, idx) for idx in range(self.num_t_factories)]

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="t", used=True)) for q in t_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="s", used=True)) for q in s_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        G.add_edges_from(
            [
                (n1, n2)
                for n1, n2 in combinations(G.nodes, 2)
                if abs(n1.row - n2.row) + abs(n1.col - n2.col) == 1
            ]
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G


class Embedded(Layout):
    """
    Lattice surgery layout based on packing logical qubits into a rectangle with ancilla patches forming gaps between them
    Without the ancilla patches, the logical qubits would be nearest neighbor
    Factories surround the main array, alternating between S and T designation
    This Layout currently cannot increase/decrease the number of factories of either type
    The inspiration for this layout was a conversation with Ben, where he described the output of the MCM compiler being nearest-neighbor connectivity
    So I wanted a Layout that could potentially be compatible with that kind of output
    """

    # TODO: figure out a way o make the number of factories configurable
    def __init__(self, input_circuit: cirq.Circuit):
        # TODO: Find the formula for this
        super().__init__(input_circuit=input_circuit, num_s_factories=0, num_t_factories=0)

    def _generate(self):
        """
        Builds a large embedded logical qubit array by starting from a nearest neighbor array and adding rows/columns of other qubit types
        """
        all_qubits = list(self.input_circuit.all_qubits())
        num_logicals = len(all_qubits)
        side_length = ceil(sqrt(num_logicals))
        filler = side_length**2 - num_logicals

        # Build a mini array that packs the logical qubits as tightly as possible in a rectangle
        # Any leftover space in the rectangle is designated as ancilla space
        stage1 = np.array([1] * num_logicals + [0] * filler).reshape((side_length, side_length))
        stage1 = np.array([row for row in stage1 if not all(row == 0)])
        stage1 = np.array([col for col in stage1.T if not all(col == 0)]).T

        # Add ancilla space between logical qubits
        stage2 = [[0] * stage1.shape[1]]
        for row in stage1:
            stage2.append(row.tolist())
            stage2.append([0] * len(row))
        stage2 = np.array(stage2)

        stage3 = [[0] * stage2.shape[0]]
        for col in stage2.T:
            stage3.append(col.tolist())
            stage3.append([0] * len(col))
        stage3 = np.array(stage3).T

        # Wrap the resulting array in factory qubits
        factory_row = np.array([2 if i % 2 else 3 for i in range(stage3.shape[1])])
        stage4 = np.vstack((factory_row, stage3, factory_row))

        factory_col = np.array([[0] + [2 if i % 2 else 3 for i in range(stage3.shape[0])] + [0]]).T
        stage5 = np.hstack((factory_col, stage4, factory_col))
        total_rows, total_cols = stage5.shape

        # Now convert that array into logical qubits, factories, and ancilla in the qubit map and layout graph
        logical_qubit_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 1
        ]
        ancilla_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 0
        ]
        # We also trim off the corners to avoid adding useless ancilla patches
        for i, j in product([0, total_rows - 1], (0, total_cols - 1)):
            ancilla_positions.remove((i, j))
        s_factory_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 2
        ]
        t_factory_positions = [
            (i, j) for i, j in product(range(total_rows), range(total_cols)) if stage5[i, j] == 3
        ]
        qubit_map = {
            qid: cirq.GridQubit(row, col)
            for qid, (row, col) in zip(sorted(all_qubits), logical_qubit_positions)
        }
        self.set_map_circuit(qubit_map=qubit_map)
        ancillas = [cirq.GridQubit(row, col) for row, col in ancilla_positions]
        s_factories = [cirq.GridQubit(row, col) for row, col in s_factory_positions]
        t_factories = [cirq.GridQubit(row, col) for row, col in t_factory_positions]

        G = nx.Graph()
        G.add_nodes_from(
            [(q, dict(patch_type="data")) for q in qubit_map.values()],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="t", used=True)) for q in t_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="factory", ftype="s", used=True)) for q in s_factories],
        )
        G.add_nodes_from(
            [(q, dict(patch_type="ancilla")) for q in ancillas],
        )
        G.add_edges_from(
            [
                (n1, n2)
                for n1, n2 in combinations(G.nodes, 2)
                if abs(n1.row - n2.row) + abs(n1.col - n2.col) == 1
            ]
        )
        self._all_factories = {node for node in G if G.nodes[node]["patch_type"] == "factory"}
        self.layout_graph = G
        self.num_s_factories = len(s_factories)
        self.num_t_factories = len(t_factories)
