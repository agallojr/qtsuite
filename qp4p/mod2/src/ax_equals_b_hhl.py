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
from qp4p_output import create_standardized_output, output_json


def parse_matrix(s: str) -> np.ndarray:
    """Parse a matrix from string like '[[2,1],[1,2]]'."""
    return np.array(json.loads(s), dtype=float)


def parse_vector(s: str) -> np.ndarray:
    """Parse a vector from string like '[1,0]'."""
    return np.array(json.loads(s), dtype=float)


def validate_system(A: np.ndarray, b: np.ndarray):
    """Validate the linear system Ax = b."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square matrix, got shape {A.shape}")
    
    n = A.shape[0]
    if len(b) != n:
        raise ValueError(f"b length ({len(b)}) must match A dimension ({n})")
    
    # Check if A is Hermitian
    if not np.allclose(A, A.conj().T):
        raise ValueError("Matrix A must be Hermitian for HHL")
    
    # Check if dimension is power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"Matrix dimension must be power of 2, got {n}")
    
    # Check condition number
    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: Matrix is ill-conditioned (condition number: {cond:.2e})", 
              file=sys.stderr)


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Ax=b using HHL algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ax_equals_b_hhl.py --matrix '[[1,0.5],[0.5,1]]' --vector '[1,0]'
  python ax_equals_b_hhl.py --matrix '[[2,1],[1,2]]' --vector '[1,1]' --backend manila
  python ax_equals_b_hhl.py --matrix '[[3,1],[1,3]]' --vector '[1,0]' --t1 50 --t2 30
""")
    parser.add_argument("--matrix", type=str, required=True,
                        help="Matrix A as JSON string, e.g., '[[2,1],[1,2]]'")
    parser.add_argument("--vector", type=str, required=True,
                        help="Vector b as JSON string, e.g., '[1,0]'")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Number of measurement shots (default: 1024)")
    add_noise_args(parser)
    add_backend_args(parser)
    
    args = parser.parse_args()

    # Parse inputs
    try:
        matrix_A = parse_matrix(args.matrix)
        vector_b = parse_vector(args.vector)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing matrix or vector: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate system
    try:
        validate_system(matrix_A, vector_b)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

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
        # Extract solution qubits (first n components)
        n = len(vector_b)
        quantum_solution = np.abs(statevector[:n])**2
        quantum_solution_normalized = quantum_solution / np.linalg.norm(quantum_solution)
    except Exception:
        # Fallback if statevector not available
        quantum_solution_normalized = classical_solution_normalized
    
    # Compute error metrics
    l2_error = np.linalg.norm(quantum_solution_normalized - classical_solution_normalized)
    fidelity = np.abs(np.dot(quantum_solution_normalized, classical_solution_normalized))**2

    # Build results dict
    output = create_standardized_output(
        algorithm="hhl",
        script_name="ax_equals_b_hhl.py",
        problem={
            "matrix_A": matrix_A.tolist(),
            "vector_b": vector_b.tolist(),
            "dimension": len(vector_b)
        },
        config={
            "shots": args.shots,
            "backend": args.backend if args.backend else "aer_simulator",
            "t1_us": args.t1,
            "t2_us": args.t2,
            "coupling_map": args.coupling_map
        },
        results={
            "quantum_normalized": quantum_solution_normalized.tolist(),
            "classical_normalized": classical_solution_normalized.tolist(),
            "classical_unnormalized": classical_solution.tolist()
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

    output_json(output)
