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

from math import pi

import cirq
import pytest

from resource_estimation import architecture, compile_ftqc, estimate
from resource_estimation import lattice_surgery_primitives as lsp
from resource_estimation.layout import Heterogenous_MovementLayout, MovementLayout


@pytest.fixture
def small_circuit() -> cirq.Circuit:
    q0, q1, q2 = cirq.LineQubit.range(3)
    return cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.T(q2))


def test_heterogenous_movement_layout_regions(small_circuit: cirq.Circuit):
    layout = Heterogenous_MovementLayout(
        small_circuit,
        num_t_factories=2,
        num_s_factories=3,
        compute_capacity=4,
    )
    graph = layout.layout_graph

    assert sum(1 for node in graph if graph.nodes[node].get("region") == "memory") == 3
    assert sum(1 for node in graph if graph.nodes[node].get("region") == "compute") == 4
    assert sum(1 for node in graph if graph.nodes[node].get("region") == "communication") >= 10
    assert (
        sum(
            1
            for node in graph
            if graph.nodes[node].get("region") == "factory"
            and graph.nodes[node].get("ftype") == "t"
        )
        == 2
    )
    assert (
        sum(
            1
            for node in graph
            if graph.nodes[node].get("region") == "factory"
            and graph.nodes[node].get("ftype") == "s"
        )
        == 3
    )
    legend_labels = {handle.get_label() for handle in layout._legend_handles()}
    assert {
        "Memory region",
        "Compute region",
        "Communication region",
        "T factory",
        "S factory",
    } <= legend_labels

    region_edges = {
        frozenset((graph.nodes[left].get("region"), graph.nodes[right].get("region")))
        for left, right in graph.edges
    }
    assert frozenset(("memory", "communication")) in region_edges
    assert frozenset(("compute", "communication")) in region_edges
    assert frozenset(("factory", "communication")) in region_edges


def test_heterogenous_movement_layout_communication_width_scales():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.Circuit(cirq.H.on_each(*qubits))

    auto_layout = Heterogenous_MovementLayout(circuit, num_t_factories=1, compute_capacity=4)
    auto_graph = auto_layout.layout_graph
    auto_comm_cols = {
        node.col for node in auto_graph if auto_graph.nodes[node].get("region") == "communication"
    }
    assert len(auto_comm_cols) == 2

    fixed_layout = Heterogenous_MovementLayout(
        circuit,
        num_t_factories=1,
        compute_capacity=4,
        communication_width=3,
    )
    fixed_graph = fixed_layout.layout_graph
    fixed_comm_cols = {
        node.col for node in fixed_graph if fixed_graph.nodes[node].get("region") == "communication"
    }
    assert len(fixed_comm_cols) == 3


def test_heterogenous_movement_layout_schedules_memory_compute_moves(
    small_circuit: cirq.Circuit,
):
    layout = Heterogenous_MovementLayout(small_circuit, num_t_factories=1, compute_capacity=4)
    graph = layout.layout_graph
    mapped_ops = list(layout.mapped_circuit.all_operations())
    move_ops = [
        op for op in mapped_ops if isinstance(op.gate, (lsp.Move, lsp.CommunicationMove))
    ]
    communication_move_ops = [
        op for op in mapped_ops if isinstance(op.gate, lsp.CommunicationMove)
    ]
    logical_ops = [
        op
        for op in mapped_ops
        if not isinstance(op.gate, (lsp.Move, lsp.CommunicationMove)) and op.gate != cirq.I
    ]

    assert len(move_ops) > 2 * sum(len(op.qubits) for op in small_circuit.all_operations())
    assert all(
        graph.nodes[qubit]["region"] == "compute" for op in logical_ops for qubit in op.qubits
    )
    assert any(graph.nodes[op.qubits[0]]["region"] == "memory" for op in move_ops)
    assert any(graph.nodes[op.qubits[1]]["region"] == "compute" for op in move_ops)
    assert any(
        graph.nodes[qubit]["region"] == "communication" for op in move_ops for qubit in op.qubits
    )
    assert communication_move_ops

    memory = next(node for node in graph if graph.nodes[node].get("region") == "memory")
    compute = next(node for node in graph if graph.nodes[node].get("region") == "compute")
    route = layout.route_through_communication(memory, compute)
    assert graph.nodes[route[0]]["region"] == "memory"
    assert graph.nodes[route[-1]]["region"] == "compute"
    assert any(graph.nodes[node]["region"] == "communication" for node in route[1:-1])
    assert all(
        graph.nodes[node]["region"] in {"memory", "compute", "communication"} for node in route
    )


def test_heterogenous_movement_layout_compiles_with_movement_only(
    small_circuit: cirq.Circuit,
):
    layout = Heterogenous_MovementLayout(small_circuit, num_t_factories=2, compute_capacity=4)
    movement_arch = architecture.DefaultMovement(idling=False, post_op_correction=False)
    primitive_circuit = compile_ftqc.ft_compile(layout=layout, arc=movement_arch, verbose=0)
    costs = estimate.ResourceEstimator(movement_arch).parallel_circuit_cost(
        primitive_circuit,
        pretty=True,
    )

    assert costs["QubitPermutationGate"] > 0
    assert any(
        isinstance(op.gate, lsp.CommunicationMove) for op in primitive_circuit.all_operations()
    )

    lattice_arch = architecture.DefaultLattice(idling=False, post_op_correction=False)
    with pytest.raises(ValueError):
        compile_ftqc.ft_compile(layout=layout, arc=lattice_arch, verbose=0)


def test_communication_move_has_separate_cost():
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    movement_arch = architecture.DefaultMovement(
        idling=False,
        post_op_correction=False,
        d=11,
        communication_move_factor=3,
    )

    standard_cost = movement_arch.move_cost(lsp.Move(zone=None).on(qubit_a, qubit_b))
    communication_cost = movement_arch.communication_move_cost(
        lsp.CommunicationMove(route_distance=1).on(qubit_a, qubit_b)
    )

    assert communication_cost["op_time"] == 3 * standard_cost["op_time"]
    assert communication_cost["gate_cost"][cirq.QubitPermutationGate] == 3


def test_communication_move_cost_uses_route_distance_instead_of_manhattan_distance():
    qubit_a, qubit_b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    movement_arch = architecture.DefaultMovement(
        idling=False,
        post_op_correction=False,
        d=11,
        communication_move_factor=3,
    )

    communication_cost = movement_arch.communication_move_cost(
        lsp.CommunicationMove(route_distance=7).on(qubit_a, qubit_b)
    )

    assert communication_cost["op_time"] == 3 * 2 * 11 * 7


def test_harq_route_moves_store_cell_distance():
    circuit = cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))
    layout = Heterogenous_MovementLayout(
        input_circuit=circuit,
        num_t_factories=1,
        num_s_factories=1,
        compute_capacity=2,
        communication_width=3,
    )
    memory = next(
        q for q, attrs in layout.layout_graph.nodes(data=True) if attrs["region"] == "memory"
    )
    compute = next(
        q for q, attrs in layout.layout_graph.nodes(data=True) if attrs["region"] == "compute"
    )

    route = layout.route_through_communication(memory, compute)
    route_moves = list(layout._moves_along_path(route).all_operations())

    assert sum(op.gate.route_distance for op in route_moves) == len(route) - 1


def _only_gate(circuit: cirq.Circuit, gate: cirq.Gate) -> cirq.Operation:
    matching_ops = [op for op in circuit.all_operations() if op.gate == gate]
    assert len(matching_ops) == 1
    return matching_ops[0]


def _route_ops_before_first_cnot(operations: list[cirq.Operation]) -> list[cirq.Operation]:
    cnot_index = next(idx for idx, op in enumerate(operations) if op.gate == cirq.CNOT)
    return [
        op
        for op in operations[:cnot_index]
        if isinstance(op.gate, (lsp.Move, lsp.CommunicationMove))
    ]


def test_harq_t_teleportation_routes_factory_to_compute():
    q0 = cirq.LineQubit(0)
    layout = Heterogenous_MovementLayout(
        input_circuit=cirq.Circuit(cirq.T(q0)),
        num_t_factories=2,
        num_s_factories=1,
        compute_capacity=1,
        communication_width=2,
    )
    graph = layout.layout_graph
    t_op = _only_gate(layout.mapped_circuit, cirq.T)

    operations = compile_ftqc.teleport_T(t_op, layout)
    route_ops = _route_ops_before_first_cnot(operations)

    assert route_ops
    assert any(isinstance(op.gate, lsp.CommunicationMove) for op in route_ops)
    assert graph.nodes[route_ops[0].qubits[0]]["region"] == "factory"
    assert graph.nodes[route_ops[-1].qubits[1]]["region"] == "compute"
    assert any(
        graph.nodes[qubit]["region"] == "communication" for op in route_ops for qubit in op.qubits
    )


def test_harq_s_teleportation_routes_factory_to_compute():
    q0 = cirq.LineQubit(0)
    layout = Heterogenous_MovementLayout(
        input_circuit=cirq.Circuit(cirq.S(q0)),
        num_t_factories=1,
        num_s_factories=2,
        compute_capacity=1,
        communication_width=2,
    )
    graph = layout.layout_graph
    s_op = _only_gate(layout.mapped_circuit, cirq.S)

    operations = compile_ftqc.teleport_S(s_op, layout)
    route_ops = _route_ops_before_first_cnot(operations)

    assert route_ops
    assert any(isinstance(op.gate, lsp.CommunicationMove) for op in route_ops)
    assert graph.nodes[route_ops[0].qubits[0]]["region"] == "factory"
    assert graph.nodes[route_ops[-1].qubits[1]]["region"] == "compute"
    assert any(
        graph.nodes[qubit]["region"] == "communication" for op in route_ops for qubit in op.qubits
    )


def test_movement_layout_t_teleportation_keeps_direct_factory_cnot():
    q0 = cirq.LineQubit(0)
    layout = MovementLayout(input_circuit=cirq.Circuit(cirq.T(q0)), num_t_factories=1)
    t_op = _only_gate(layout.mapped_circuit, cirq.T)

    operations = compile_ftqc.teleport_T(t_op, layout)

    assert not any(isinstance(op.gate, lsp.CommunicationMove) for op in operations)
    assert not any(isinstance(op.gate, lsp.Move) for op in operations)
    assert [op.gate for op in operations] == [
        lsp.Cultivate(pi / 4),
        cirq.CNOT,
        cirq.MeasurementGate(1, key=""),
        cirq.S,
        cirq.ResetChannel(),
    ]
