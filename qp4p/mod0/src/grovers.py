"""
Grover's algorithm demonstration

Sample execution:
    python src/grovers.py
    python src/grovers.py --targets 101 110 --shots 2048
    python src/grovers.py --backend jakarta --no-display
"""

import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import AmplificationProblem, Grover

from qp4p_circuit import run_circuit, BASIS_GATES

# *****************************************************************************

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Grover's algorithm demonstration")
    parser.add_argument("--targets", type=str, nargs='+', default=["10111"],
                        help="Target state(s) to search for (default: 10111)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of Grover iterations (default: computed optimally)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Number of shots for noisy simulation (default: 1024)")
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in microseconds (default: no noise)")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in microseconds (default: no noise)")
    parser.add_argument("--backend", type=str, default=None,
                        help="Fake backend name (e.g., 'manila', 'jakarta')")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable graphical display of circuit and plots")
    args = parser.parse_args()
    display = not args.no_display

    marked_states = args.targets
    
    # Validate all targets have same length
    if len(set(len(s) for s in marked_states)) > 1:
        raise ValueError("All target states must have the same length")

    # 1. Define the "Oracle" - the state(s) you want to find
    # The oracle "marks" target states by applying a phase flip (-1) to them.
    # For multiple targets, create a summed superposition of marked states.
    oracle_vector = sum((Statevector.from_label(s) for s in marked_states), 
                        Statevector.from_label('0' * len(marked_states[0])) * 0)
    oracle = oracle_vector / np.linalg.norm(oracle_vector.data)  # Normalize

    # 2. Define the Amplification Problem
    problem = AmplificationProblem(oracle, is_good_state=marked_states)

    # 3. Calculate ideal # of iterations
    # For N=2^n states with M marked states: optimal iterations ≈ π/4 * √(N/M)
    num_qubits = len(marked_states[0])
    num_states = 2 ** num_qubits
    num_marked = len(marked_states)
    optimal_iterations = round(math.pi / 4 * math.sqrt(num_states / num_marked))
    iterations = args.iterations if args.iterations is not None else optimal_iterations

    # 4. Instantiate the Grover class with explicit iterations
    sampler = StatevectorSampler()
    grover = Grover(sampler=sampler, iterations=iterations)

    # 5. Construct the circuit
    grover_circuit = grover.construct_circuit(problem)

    # 6. Build results dict (will be printed as JSON at end)
    results = {
        "circuit_stats": {
            "qubits": grover_circuit.num_qubits,
            "depth": grover_circuit.depth(),
            "gate_counts": dict(grover_circuit.count_ops()),
            "iterations": iterations,
            "optimal_iterations": optimal_iterations
        },
        "target_states": marked_states
    }

    # 7. We want an illustrative look under the hood.
    # Build probability snapshots for each iteration (only if display is enabled)
    # If noise is specified, use shots-based simulation; otherwise use ideal statevector
    use_noise = args.t1 is not None or args.t2 is not None or args.backend is not None

    if display:
        target_indices = [int(s, 2) for s in marked_states]
        probabilities_by_iter = []

        for num_iter in range(iterations + 1):
            if num_iter == 0:
                # Just uniform superposition (H gates on all qubits)
                qc_iter = QuantumCircuit(num_qubits)
                qc_iter.h(range(num_qubits))
            else:
                grover_iter = Grover(sampler=sampler, iterations=num_iter)
                qc_iter = grover_iter.construct_circuit(problem)
            
            if use_noise:
                # Noisy simulation: transpile to basis gates, add measurements, run with noise
                qc_transpiled = transpile(qc_iter, basis_gates=BASIS_GATES)
                qc_transpiled.measure_all()
                run_result = run_circuit(qc_transpiled, shots=args.shots, t1=args.t1, t2=args.t2, backend=args.backend)
                # Convert counts to probability distribution
                probs = np.zeros(num_states)
                for bitstring, count in run_result["counts"].items():
                    idx = int(bitstring, 2)
                    probs[idx] = count / args.shots
            else:
                # Ideal simulation: use statevector
                sv = Statevector.from_instruction(qc_iter)
                probs = np.abs(sv.data) ** 2
            
            probabilities_by_iter.append(probs)

        # 8. Create combined figure with circuit and iteration plots
        fig = plt.figure(figsize=(max(16, 4 * (iterations + 1)), 10))
        gs = GridSpec(2, iterations + 1, figure=fig, height_ratios=[1.2, 1])

        # Top row: circuit diagram spanning all columns
        ax_circuit = fig.add_subplot(gs[0, :])
        decomposed = grover_circuit.decompose().decompose()
        decomposed.draw('mpl', ax=ax_circuit, fold=-1)
        ax_circuit.set_title("Grover Circuit (decomposed)")

        # Bottom row: probability distributions for each iteration
        for i, probs in enumerate(probabilities_by_iter):
            ax = fig.add_subplot(gs[1, i])
            x = range(len(probs))
            colors = ['red' if idx in target_indices else 'steelblue' for idx in x]
            ax.bar(x, probs, color=colors)
            ax.set_title(f"After {i} iter")
            ax.set_xlabel("State")
            ax.set_ylabel("Prob" if i == 0 else "")
            ax.set_ylim(0, 1)
            total_target_prob = sum(probs[idx] for idx in target_indices)
            ax.text(0.95, 0.95, f"P={total_target_prob:.2f}", 
                    transform=ax.transAxes, ha='right', va='top', color='red', fontsize=9)

        target_labels = ", ".join(f"|{s}⟩" for s in marked_states)
        noise_label = f" (T1={args.t1}µs, T2={args.t2}µs)" if use_noise else " (ideal)"
        fig.suptitle(f"Grover's Algorithm: {target_labels}{noise_label}", fontsize=14)
        plt.tight_layout()
        plt.show()

    # 9. Now actually run the algorithm with optional noise
    # Transpile to basis gates and add measurements for shots-based execution
    grover_with_meas = transpile(grover_circuit, basis_gates=BASIS_GATES)
    grover_with_meas.measure_all()
    
    run_result = run_circuit(grover_with_meas, shots=args.shots, t1=args.t1, t2=args.t2, backend=args.backend)
    counts = run_result["counts"]
    
    # Find top measurement
    top_measurement = max(counts, key=counts.get)
    is_success = top_measurement in marked_states

    # 10. Add run results to output
    results["run"] = {
        "top_measurement": top_measurement,
        "search_successful": is_success,
        "shots": args.shots,
        "top_counts": {
            state: {"count": count, "percent": round(100 * count / args.shots, 1)}
            for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    }
    results["run"]["t1_us"] = args.t1
    results["run"]["t2_us"] = args.t2
    results["transpiled_stats"] = run_result["transpiled_stats"]
    results["backend_info"] = json.dumps(run_result["backend_info"], separators=(',', ':')) if run_result["backend_info"] else None

    # 11. Print results as JSON
    print(json.dumps(results, indent=2))
