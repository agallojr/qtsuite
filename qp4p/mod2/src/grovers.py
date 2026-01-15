"""
Grover's algorithm demonstration

Sample execution:
    python src/grovers.py
    python src/grovers.py --targets 101 110 --shots 2048
    python src/grovers.py --backend jakarta
"""

import argparse
import json
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import AmplificationProblem, Grover

from qp4p_circuit import run_circuit, BASIS_GATES
from qp4p_args import add_standard_quantum_args
from qp4p_output import create_standardized_output, output_json

# *****************************************************************************

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Grover's algorithm demonstration")
    parser.add_argument("--targets", type=str, nargs='+', default=["10111"],
                        help="Target state(s) to search for (default: 10111)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of Grover iterations (default: computed optimally)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of qubits (default: 3)")
    add_standard_quantum_args(parser, include_shots=True)
    args = parser.parse_args()

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
        "target_states": marked_states,
        "num_qubits": num_qubits,
        "num_states": num_states,
        "num_marked": num_marked
    }

    # 7. Build probability snapshots for each iteration for visualization
    # If noise is specified, use shots-based simulation; otherwise use ideal statevector
    use_noise = args.t1 is not None or args.t2 is not None or args.backend is not None
    
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
            run_result = run_circuit(qc_transpiled, shots=args.shots, t1=args.t1, t2=args.t2, 
                                   backend=args.backend, coupling_map=args.coupling_map)
            # Convert counts to probability distribution
            probs = np.zeros(num_states)
            for bitstring, count in run_result["counts"].items():
                idx = int(bitstring, 2)
                probs[idx] = count / args.shots
        else:
            # Ideal simulation: use statevector
            sv = Statevector.from_instruction(qc_iter)
            probs = np.abs(sv.data) ** 2
        
        probabilities_by_iter.append(probs.tolist())
    
    # Store visualization data
    results["visualization_data"] = {
        "probabilities_by_iteration": probabilities_by_iter,
        "target_indices": target_indices,
        "use_noise": use_noise
    }

    # 8. Now actually run the algorithm with optional noise
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
    output = create_standardized_output(
        algorithm="grover",
        script_name="grovers.py",
        problem={
            "num_qubits": num_qubits,
            "num_states": num_states,
            "num_marked": num_marked,
            "target_states": marked_states
        },
        config={
            "iterations": iterations,
            "shots": args.shots,
            "t1_us": args.t1,
            "t2_us": args.t2
        },
        results={
            "search_successful": is_success,
            "top_measurement": top_measurement,
            "top_counts": {
                state: {"count": count, "percent": round(100 * count / args.shots, 1)}
                for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        },
        circuit_info=results["circuit_stats"],
        backend_info=json.dumps(run_result["backend_info"], separators=(',', ':')) if run_result["backend_info"] else None,
        visualization_data=results.get("visualization_data")
    )
    
    output_json(output)
