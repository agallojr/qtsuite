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
import os
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

# Disable Qrisp verbose output before importing
os.environ['QRISP_VERBOSE'] = '0'

from qrisp.algorithms.cks import CKS, inner_CKS
from qrisp.jasp import terminal_sampling

from qp4p_output import create_standardized_output, output_error
from qp4p_linear_system import add_linear_system_args, get_linear_system




# Matrix parsing and validation moved to qp4p_linear_system helper


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
        # Check for NaN or invalid probabilities
        if np.isnan(prob) or np.isinf(prob):
            raise ValueError(f"CKS returned invalid probability (NaN/Inf). "
                           f"Try increasing epsilon or using a better-conditioned matrix.")
        
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
  python ax_equals_b_cks_qrisp_est.py --size 2
  python ax_equals_b_cks_qrisp_est.py --size 2 --seed 42
  python ax_equals_b_cks_qrisp_est.py --size 2 --tridiag
  python ax_equals_b_cks_qrisp_est.py --matrix '[[2,1],[1,2]]' --vector '[1,0]'
""")
    add_linear_system_args(parser)
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Target precision (default: 0.01)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Number of measurement shots (default: 1024)")
    
    args = parser.parse_args()

    # Get linear system (require Hermitian and power of 2 for CKS)
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
    
    n = matrix_A.shape[0]
    eigenvalues = np.linalg.eigvals(matrix_A)
    condition_number = sys_metadata['condition_number']

    # Suppress all Qrisp output (progress bars, etc.)
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            # Get circuit information before running
            try:
                circuit_info = get_cks_circuit_info(matrix_A, vector_b, args.epsilon)
            except Exception:
                circuit_info = {
                    "num_qubits": None,
                    "num_clbits": None,
                    "depth": None,
                    "num_gates": None,
                    "gate_counts": {}
                }

            # Run CKS algorithm
            counts = run_cks_algorithm(matrix_A, vector_b, args.epsilon, args.shots)

    n_qubits = int(np.log2(len(vector_b)))
    quantum_solution = extract_solution_from_counts(counts, n_qubits)

    # Compute classical solution for comparison
    classical_solution = np.linalg.solve(matrix_A, vector_b)
    classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)

    # Keep raw quantum solution before normalization (note: extraction function already normalizes)
    quantum_solution_raw = quantum_solution.copy()
    
    # Compute error metrics on normalized solutions
    l2_error = np.linalg.norm(quantum_solution - classical_solution_normalized)
    fidelity = np.abs(np.dot(quantum_solution, classical_solution_normalized))**2

    # Build standardized output
    output = create_standardized_output(
        algorithm="cks",
        script_name="ax_equals_b_cks_qrisp_est.py",
        problem={
            "matrix": matrix_A.tolist(),
            "rhs": vector_b.tolist(),
            "dimension": n,
            "condition_number": float(condition_number),
            "eigenvalues": [float(e) for e in eigenvalues]
        },
        config={
            "epsilon": args.epsilon,
            "shots": args.shots
        },
        results={
            "classical_solution": classical_solution.tolist(),
            "classical_solution_normalized": classical_solution_normalized.tolist(),
            "quantum_solution": quantum_solution_raw.tolist(),
            "quantum_solution_normalized": quantum_solution.tolist()
        },
        metrics={
            "l2_error": float(l2_error),
            "fidelity": float(fidelity)
        },
        circuit_info=circuit_info
    )

    print(json.dumps(output, indent=2))
