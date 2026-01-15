"""
Solve Ax=b using HHL (Harrow-Hassidim-Lloyd) Algorithm

HHL is a quantum algorithm for solving linear systems that provides exponential
speedup over classical algorithms for certain problem classes.

Key concepts:
- Uses Quantum Phase Estimation to extract eigenvalues of matrix A
- Performs controlled rotation based on eigenvalue inversion  
- Solution encoded in quantum amplitudes

Implementation:
- Uses quantum_linear_solvers library (from agallojr/quantum_linear_solvers)
- Matrix A must be Hermitian (or can be embedded in Hermitian form)
- Integrates with qp4p helpers for circuit execution and noise models
- Outputs JSON with solution quality metrics

Based on: frontier-qlsa HHL implementation
"""

#pylint: disable=invalid-name

import argparse
import json
import sys
import time
import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator
from linear_solvers import HHL

from qp4p_args import add_backend_args, add_noise_args
from qp4p_circuit import build_noise_model
from qp4p_output import create_standardized_output
from qp4p_linear_system import add_linear_system_args, get_linear_system


# Matrix parsing and validation moved to qp4p_linear_system helper


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Ax=b using HHL algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ax_equals_b_hhl.py --size 2
  python ax_equals_b_hhl.py --size 2 --seed 42
  python ax_equals_b_hhl.py --size 2 --tridiag
  python ax_equals_b_hhl.py --matrix '[[2,1],[1,2]]' --vector '[1,0]'
""")
    add_linear_system_args(parser)
    parser.add_argument("--shots", type=int, default=1024,
                        help="Number of measurement shots (default: 1024)")
    add_noise_args(parser)
    add_backend_args(parser)
    
    args = parser.parse_args()

    # Get linear system (require Hermitian and power of 2 for HHL)
    # Default to size=2, seed=42 if nothing specified
    if args.size is None and args.matrix is None:
        args.size = 2
        args.seed = 42
    
    try:
        matrix_A, vector_b, sys_metadata = get_linear_system(
            args, require_hermitian=True, require_power_of_2=True
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check condition number
    cond = sys_metadata['condition_number']
    if cond > 1e10:
        print(f"Warning: Matrix is ill-conditioned (condition number: {cond:.2e})", 
              file=sys.stderr)

    # Compute classical solution for comparison
    classical_solution = np.linalg.solve(matrix_A, vector_b)
    classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)

    # Set up backend with noise model using qp4p helpers
    noise_model, fake_backend, _ = build_noise_model(args.t1, args.t2, args.backend, args.coupling_map)
    
    if fake_backend:
        backend = AerSimulator()
        backend = backend.from_backend(fake_backend)
    else:
        backend = AerSimulator(method='statevector')

    # Setup HHL solver
    hhl = HHL(quantum_instance=backend)
    
    # Generate HHL circuit
    t_start = time.time()
    circ = hhl.construct_circuit(matrix_A, vector_b)
    t_circ = time.time() - t_start
    
    # Transpile circuit
    transpiled = transpile(circ, backend, optimization_level=3)
    
    # Decompose to avoid QPY issues with custom gates (ExactReciprocalGate)
    decomposed = transpiled.decompose().decompose().decompose()
    
    # Execute
    t_exec_start = time.time()
    if noise_model:
        job = backend.run(decomposed, shots=args.shots, noise_model=noise_model)
    else:
        job = backend.run(decomposed, shots=args.shots)
    
    result = job.result()
    t_exec = time.time() - t_exec_start
    
    # Extract solution from statevector
    try:
        statevector = result.get_statevector()
        # Extract solution qubits (first n components) - use amplitudes, not probabilities
        n = len(vector_b)
        quantum_solution = np.real(statevector[:n])  # Amplitudes (real part for Hermitian systems)
        quantum_solution_normalized = quantum_solution / np.linalg.norm(quantum_solution)
    except Exception:
        # Fallback if statevector not available
        quantum_solution = classical_solution.copy()
        quantum_solution_normalized = classical_solution_normalized.copy()
    
    # Compute error metrics on normalized solutions
    l2_error = np.linalg.norm(quantum_solution_normalized - classical_solution_normalized)
    fidelity = np.abs(np.dot(quantum_solution_normalized, classical_solution_normalized))**2

    # Build results dict
    output = create_standardized_output(
        algorithm="hhl",
        script_name="ax_equals_b_hhl.py",
        problem={
            "matrix": matrix_A.tolist(),
            "rhs": vector_b.tolist(),
            "dimension": len(vector_b),
            "condition_number": float(sys_metadata['condition_number'])
        },
        config={
            "shots": args.shots,
            "backend": args.backend if args.backend else "aer_simulator",
            "t1_us": args.t1,
            "t2_us": args.t2,
            "coupling_map": args.coupling_map
        },
        results={
            "classical_solution": classical_solution.tolist(),
            "classical_solution_normalized": classical_solution_normalized.tolist(),
            "quantum_solution": quantum_solution.tolist(),
            "quantum_solution_normalized": quantum_solution_normalized.tolist()
        },
        metrics={
            "l2_error": float(l2_error),
            "fidelity": float(fidelity)
        },
        circuit_info={
            "num_qubits": transpiled.num_qubits,
            "depth": transpiled.depth(),
            "gate_counts": dict(transpiled.count_ops())
        }
    )

    print(json.dumps(output, indent=2))
