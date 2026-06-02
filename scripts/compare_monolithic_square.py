#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/harq-resource-superstaq-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/harq-resource-superstaq-cache")

import cirq
import numpy as np

import resource_estimation as res


@dataclass(frozen=True)
class Scenario:
    name: str
    assumption: str
    layout_factory: Callable[[cirq.Circuit], res.layout.Layout]
    architecture_factory: Callable[[], res.architecture.Architecture]


def _notebook_circuit(num_qubits: int, depth: int, seed: int) -> cirq.Circuit:
    rng = np.random.default_rng(seed)
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    for layer in range(depth):
        circuit.append(cirq.H.on_each(*qubits))
        circuit.append(cirq.rz(float(rng.uniform(-np.pi, np.pi))).on(q) for q in qubits)
        start = layer % 2
        circuit.append(
            cirq.CNOT(qubits[i], qubits[i + 1])
            for i in range(start, num_qubits - 1, 2)
        )
    return circuit


def _synthesize(circuit: cirq.Circuit, fidelity: float) -> tuple[cirq.Circuit, cirq.Circuit, float]:
    cliff_rz_circuit = res.cliff_rz.compile_cliff_rz(circuit)
    rz_count = sum(op.gate in cirq.GateFamily(cirq.Rz) for op in cliff_rz_circuit.all_operations())
    eps = 1 - fidelity ** (1 / rz_count) if rz_count else 0.0
    synthesized_circuit = res.clifford_t.compile_cirq_to_clifford_t(
        cliff_rz_circuit,
        eps=eps,
        verbose=False,
    )
    synthesized_circuit = synthesized_circuit.transform_qubits(
        {qubit: cirq.LineQubit(i) for i, qubit in enumerate(sorted(synthesized_circuit.all_qubits()))}
    )
    return cliff_rz_circuit, synthesized_circuit, eps


def _build_scenarios(args: argparse.Namespace) -> dict[str, Scenario]:
    movement_kwargs = dict(
        d=args.distance,
        idling=args.idling,
        post_op_correction=args.post_op_correction,
        syndrome_rounds=args.syndrome_rounds,
        cultivation_repetition=args.cultivation_repetition,
        cultivation_fault_distance=args.cultivation_fault_distance,
        communication_move_factor=args.communication_move_factor,
    )
    lattice_kwargs = dict(
        d=args.distance,
        idling=args.idling,
        post_op_correction=args.post_op_correction,
        syndrome_rounds=args.syndrome_rounds,
        cultivation_repetition=args.cultivation_repetition,
        cultivation_fault_distance=args.cultivation_fault_distance,
    )

    scenarios = [
        Scenario(
            name="monolithic_ssm",
            assumption=(
                "DefaultMovement + MovementLayout; transversal CNOTs, T factories, "
                "and neutral-atom move/readout timings."
            ),
            layout_factory=lambda circuit: res.layout.MovementLayout(
                input_circuit=circuit,
                num_t_factories=args.num_t_factories,
            ),
            architecture_factory=lambda: res.architecture.DefaultMovement(**movement_kwargs),
        ),
        Scenario(
            name="monolithic_ssoq",
            assumption=(
                "Superconductor + FactorySandwich; same T/S factory counts as Square, "
                "local lattice-surgery CNOT routing, and superconducting timings."
            ),
            layout_factory=lambda circuit: res.layout.FactorySandwich(
                input_circuit=circuit,
                num_t_factories=args.num_t_factories,
                num_s_factories=args.num_s_factories,
            ),
            architecture_factory=lambda: res.architecture.Superconductor(**lattice_kwargs),
        ),
        Scenario(
            name="square_heterogeneous",
            assumption=(
                "Square regional layout; SSM memory/bus, SSOQ compute/factories, "
                "explicit ModalityTransfer crossings."
            ),
            layout_factory=lambda circuit: res.layout.Square(
                input_circuit=circuit,
                num_t_factories=args.num_t_factories,
                num_s_factories=args.num_s_factories,
                compute_capacity=args.compute_capacity,
            ),
            architecture_factory=lambda: res.architecture.RegionalArchitecture.square_default(
                d=args.distance,
                idling=args.idling,
                post_op_correction=args.post_op_correction,
                syndrome_rounds=args.syndrome_rounds,
                cultivation_repetition=args.cultivation_repetition,
                cultivation_fault_distance=args.cultivation_fault_distance,
                communication_move_factor=args.communication_move_factor,
                transfer_penalty=args.transfer_penalty,
            ),
        ),
    ]
    return {scenario.name: scenario for scenario in scenarios}


def _estimate_scenario(
    scenario: Scenario,
    synthesized_circuit: cirq.Circuit,
    *,
    verbose: bool,
) -> dict[str, object]:
    if verbose:
        print(f"Estimating {scenario.name}...", file=sys.stderr, flush=True)
    layout = scenario.layout_factory(synthesized_circuit)
    architecture = scenario.architecture_factory()
    primitive_circuit = res.compile_ftqc.ft_compile(layout=layout, arc=architecture, verbose=False)
    estimator = res.estimate.ResourceEstimator(arc=architecture, layout=layout)
    primitive_counts = Counter(type(op.gate).__name__ for op in primitive_circuit.all_operations())

    return {
        "scenario": scenario.name,
        "architecture": str(architecture),
        "logical_qubits": cirq.num_qubits(synthesized_circuit),
        "layout_nodes": len(layout.layout_graph.nodes),
        "primitive_ops": len(list(primitive_circuit.all_operations())),
        "parallel_time_us": estimator.parallel_circuit_time(primitive_circuit),
        "serial_time_us": estimator.serial_circuit_time(primitive_circuit),
        "physical_qubits": estimator.physical_qubits(primitive_circuit),
        "moves": primitive_counts.get("Move", 0),
        "communication_moves": primitive_counts.get("CommunicationMove", 0),
        "transfers": primitive_counts.get("ModalityTransfer", 0),
        "merges": primitive_counts.get("Merge", 0),
        "splits": primitive_counts.get("Split", 0),
        "cultivations": primitive_counts.get("Cultivate", 0),
        "assumption": scenario.assumption,
    }


def _format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def _print_table(rows: list[dict[str, object]]) -> None:
    columns = [
        "scenario",
        "parallel_time_us",
        "serial_time_us",
        "physical_qubits",
        "layout_nodes",
        "primitive_ops",
        "moves",
        "communication_moves",
        "transfers",
        "merges",
        "splits",
        "cultivations",
    ]
    widths = {
        column: max(len(column), *(len(_format_float(row[column])) for row in rows))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            " | ".join(
                _format_float(row[column]).rjust(widths[column])
                if column != "scenario"
                else str(row[column]).ljust(widths[column])
                for column in columns
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare monolithic and Square heterogeneous estimates on the same "
            "synthetic circuit and synthesis output."
        )
    )
    parser.add_argument("--num-qubits", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fidelity", type=float, default=0.99)
    parser.add_argument("--num-t-factories", type=int, default=5)
    parser.add_argument("--num-s-factories", type=int, default=5)
    parser.add_argument("--compute-capacity", type=int, default=4)
    parser.add_argument("--distance", type=int, default=11)
    parser.add_argument("--syndrome-rounds", type=int, default=1)
    parser.add_argument("--cultivation-repetition", type=int, default=5)
    parser.add_argument("--cultivation-fault-distance", type=int, default=3)
    parser.add_argument("--communication-move-factor", type=float, default=3.0)
    parser.add_argument("--transfer-penalty", type=float, default=1.0)
    parser.add_argument("--idling", action="store_true")
    parser.add_argument("--no-post-op-correction", dest="post_op_correction", action="store_false")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("monolithic_ssm", "monolithic_ssoq", "square_heterogeneous"),
        default=("monolithic_ssm", "monolithic_ssoq", "square_heterogeneous"),
        help="Subset of scenarios to estimate.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide scenario progress messages.")
    parser.set_defaults(post_op_correction=True)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    circuit = _notebook_circuit(args.num_qubits, args.depth, args.seed)
    cliff_rz_circuit, synthesized_circuit, eps = _synthesize(circuit, args.fidelity)

    print("Input")
    print(f"  logical circuit: {args.num_qubits} qubits, depth {args.depth}, seed {args.seed}")
    print(f"  Clifford+Rz moments: {len(cliff_rz_circuit)}")
    print(f"  Clifford+T moments: {len(synthesized_circuit)}")
    print(f"  synthesis eps per Rz: {eps:.6e}")
    print()

    scenarios = _build_scenarios(args)
    rows = [
        _estimate_scenario(
            scenarios[scenario_name],
            synthesized_circuit,
            verbose=not args.quiet,
        )
        for scenario_name in args.scenarios
    ]
    _print_table(rows)
    print()
    print("Assumptions")
    for row in rows:
        print(f"  {row['scenario']}: {row['assumption']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
