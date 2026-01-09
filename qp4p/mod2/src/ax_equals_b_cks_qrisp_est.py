"""
Solve Ax=b using CKS (Childs-Kothari-Somma) Algorithm with Qrisp

CKS is a quantum algorithm for solving linear systems that provides better
precision scaling compared to HHL, with complexity O(log(N)s κ² polylog(sκ/ε)).

Key concepts:
- Uses Chebyshev polynomial approximation of 1/x
- Linear combination of unitaries (LCU) with qubitization
- Repeat-Until-Success (RUS) protocol for post-selection
- Better precision scaling: polylog(1/ε) vs polynomial in HHL

Implementation:
- Uses Qrisp library for high-level quantum programming
- Matrix A must be Hermitian
- Automatically constructs block-encoding via Pauli decomposition
- RUS protocol ensures valid solution through post-selection

Based on: https://qrisp.eu/reference/Algorithms/CKS.html
Paper: https://arxiv.org/abs/1511.02306
"""

#pylint: disable=invalid-name

import argparse
import json
import sys
import numpy as np

from qrisp.algorithms.cks import CKS, inner_CKS
from qrisp.jasp import terminal_sampling




def parse_matrix(s: str) -> np.ndarray:
    """Parse a matrix from string like '[[2,1],[1,2]]'."""
    return np.array(json.loads(s), dtype=float)


def parse_vector(s: str) -> np.ndarray:
    """Parse a vector from string like '[1,0]'."""
    return np.array(json.loads(s), dtype=float)


def validate_hermitian(matrix_A, tol=1e-10):
    """
    Validates that matrix A is Hermitian.
    
    Args:
        matrix_A: Input matrix
        tol: Tolerance for Hermitian check
    
    Returns:
        True if Hermitian, False otherwise
    """
    return np.allclose(matrix_A, matrix_A.conj().T, atol=tol)


def extract_solution_from_counts(counts, n_qubits_matrix):
    """
    Extracts the CKS solution vector from measurement counts.
    
    The CKS algorithm encodes the solution in quantum amplitudes.
    We extract amplitudes from measurement counts (amplitude = sqrt(probability)).
    
    Args:
        counts: Dictionary mapping measurement outcomes to counts
        n_qubits_matrix: Number of qubits encoding the matrix dimension
    
    Returns:
        Normalized quantum solution vector as numpy array
    """
    n_solution = 2 ** n_qubits_matrix
    total_shots = sum(counts.values())
    quantum_solution = np.zeros(n_solution)

    # Extract solution from measurement counts
    # The solution qubits are the first n_qubits_matrix qubits
    for bitstring, count in counts.items():
        # Convert bitstring to state index
        if isinstance(bitstring, str):
            if bitstring.startswith('0x'):
                state = int(bitstring, 16)
            else:
                state = int(bitstring, 2)
        elif isinstance(bitstring, int):
            state = bitstring
        else:
            continue
        
        # Extract the solution bits (rightmost n_qubits_matrix bits)
        solution_bits = state & ((1 << n_qubits_matrix) - 1)
        
        if solution_bits < n_solution:
            # Amplitude is sqrt of probability
            prob = count / total_shots
            quantum_solution[solution_bits] += np.sqrt(prob)

    # Filter near-zero components
    quantum_solution[np.abs(quantum_solution) < 1e-10] = 0

    # Normalize
    if np.linalg.norm(quantum_solution) > 0:
        quantum_solution = quantum_solution / np.linalg.norm(quantum_solution)

    return quantum_solution


def get_cks_circuit_info(matrix_A, vector_b, epsilon):
    """
    Extracts circuit information from CKS using inner_CKS.
    
    Note: This creates the full circuit structure which may be very large.
    We only extract statistics, not run the circuit.
    
    Args:
        matrix_A: Hermitian matrix
        vector_b: Input vector
        epsilon: Target precision
    
    Returns:
        Dictionary with circuit information
    """
    # Use inner_CKS to get the circuit structure without RUS
    operand, in_case, out_case = inner_CKS(matrix_A, vector_b, epsilon)
    
    # Get quantum session
    qs = operand.qs()
    
    # Get circuit statistics
    circuit_info = {
        "num_qubits": len(qs.qubits),
        "num_clbits": len(qs.clbits),
        "depth": qs.depth(),
        "num_gates": len(qs.data),
        "operand_qubits": operand.size,
        "in_case_qubits": in_case.size,
        "out_case_qubits": out_case.size
    }
    
    # Try to get gate counts from Qiskit conversion
    try:
        qiskit_circuit = qs.to_qiskit()
        circuit_info["gate_counts"] = dict(qiskit_circuit.count_ops())
    except Exception:
        circuit_info["gate_counts"] = {}
    
    return circuit_info


def run_cks_algorithm(matrix_A, vector_b, epsilon, shots):
    """
    Runs CKS algorithm with terminal_sampling.
    
    Args:
        matrix_A: Hermitian matrix
        vector_b: Input vector
        epsilon: Target precision
        shots: Number of shots
    
    Returns:
        Dictionary with counts
    """
    @terminal_sampling
    def cks_solve():
        x = CKS(matrix_A, vector_b, epsilon)
        return x
    
    prob_dict = cks_solve()
    
    counts = {}
    for state, prob in prob_dict.items():
        count = int(prob * shots)
        if count > 0:
            if isinstance(state, (int, float)):
                state_int = int(round(state))
                counts[f"{state_int:0{int(np.log2(len(vector_b)))}b}"] = count
    
    return counts


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Ax=b using CKS algorithm with Qrisp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ax_equals_b_cks_qrisp_est.py --matrix '[[3,1],[1,3]]' --vector '[1,1]'
  python ax_equals_b_cks_qrisp_est.py --matrix '[[2,1],[1,2]]' --vector '[1,0]' --epsilon 0.01
  python ax_equals_b_cks_qrisp_est.py --matrix '[[4,1],[1,4]]' --vector '[1,1]' --shots 2048
""")
    parser.add_argument("--matrix", type=str, required=True,
                        help="Matrix A as JSON string, e.g., '[[2,1],[1,2]]'")
    parser.add_argument("--vector", type=str, required=True,
                        help="Vector b as JSON string, e.g., '[1,0]'")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Target precision (default: 0.01)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Number of measurement shots (default: 1024)")
    
    args = parser.parse_args()

    # Parse inputs
    try:
        matrix_A = parse_matrix(args.matrix)
        vector_b = parse_vector(args.vector)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing matrix or vector: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate dimensions
    if matrix_A.ndim != 2 or matrix_A.shape[0] != matrix_A.shape[1]:
        print(f"Error: Matrix must be square, got shape {matrix_A.shape}", file=sys.stderr)
        sys.exit(1)
    
    n = matrix_A.shape[0]
    if len(vector_b) != n:
        print(f"Error: Vector length ({len(vector_b)}) must match matrix dimension ({n})", 
              file=sys.stderr)
        sys.exit(1)
    
    # Check if dimension is power of 2
    if n & (n - 1) != 0:
        print(f"Error: Matrix dimension must be power of 2, got {n}", file=sys.stderr)
        sys.exit(1)

    # Validate Hermitian
    if not validate_hermitian(matrix_A):
        print("Error: Matrix A must be Hermitian", file=sys.stderr)
        sys.exit(1)

    # Compute matrix properties
    eigenvalues = np.linalg.eigvals(matrix_A)
    condition_number = np.linalg.cond(matrix_A)

    # Get circuit information before running
    try:
        circuit_info = get_cks_circuit_info(matrix_A, vector_b, args.epsilon)
    except Exception as e:
        print(f"Warning: Could not extract circuit info: {e}", file=sys.stderr)
        circuit_info = {
            "num_qubits": None,
            "num_clbits": None,
            "depth": None,
            "num_gates": None,
            "gate_counts": {}
        }

    # Run CKS algorithm
    try:
        counts = run_cks_algorithm(matrix_A, vector_b, args.epsilon, args.shots)
    except Exception as e:
        print(f"Error running CKS algorithm: {e}", file=sys.stderr)
        sys.exit(1)

    n_qubits = int(np.log2(len(vector_b)))
    quantum_solution = extract_solution_from_counts(counts, n_qubits)

    # Compute classical solution for comparison
    classical_solution = np.linalg.solve(matrix_A, vector_b)
    classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)

    # Compute error metrics
    l2_error = np.linalg.norm(quantum_solution - classical_solution_normalized)
    fidelity = np.abs(np.dot(quantum_solution, classical_solution_normalized))**2

    # Build results dict
    results = {
        "problem": {
            "matrix_A": matrix_A.tolist(),
            "vector_b": vector_b.tolist(),
            "dimension": n
        },
        "matrix_properties": {
            "eigenvalues": [float(e) for e in eigenvalues],
            "condition_number": float(condition_number)
        },
        "cks_config": {
            "epsilon": args.epsilon,
            "shots": args.shots
        },
        "circuit_info": circuit_info,
        "solutions": {
            "quantum_normalized": quantum_solution.tolist(),
            "classical_normalized": classical_solution_normalized.tolist(),
            "classical_unnormalized": classical_solution.tolist()
        },
        "metrics": {
            "l2_error": float(l2_error),
            "fidelity": float(fidelity)
        }
    }

    print(json.dumps(results, indent=2))
