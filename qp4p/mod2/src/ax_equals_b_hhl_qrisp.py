"""
Solve Ax=b using HHL (Harrow-Hassidim-Lloyd) Algorithm with Qrisp

HHL is a quantum algorithm for solving linear systems that provides exponential
speedup over classical algorithms for certain problem classes.

Key concepts:
- Uses Quantum Phase Estimation to extract eigenvalues of matrix A
- Performs controlled rotation based on eigenvalue inversion
- Solution encoded in quantum amplitudes (requires amplitude estimation to extract)

Implementation:
- Uses Qrisp library for high-level quantum programming
- Matrix A must be Hermitian (or can be embedded in Hermitian form)
- Eigenvalues are scaled to powers of 2 for bit-reversal inversion
- QPE precision determines accuracy of eigenvalue estimation

Based on: https://qrisp.eu/general/tutorial/HHL.html
"""

#pylint: disable=invalid-name

import argparse
import json
import sys
import os
import numpy as np
from contextlib import redirect_stdout

# Disable Qrisp verbose output before importing
os.environ['QRISP_VERBOSE'] = '0'

from qrisp import QuantumFloat, QPE, prepare, cx, swap, invert
from qrisp.operators import QubitOperator
from qiskit_aer import Aer
from qiskit import transpile


from qp4p_args import add_backend_args, add_noise_args
from qp4p_circuit import build_noise_model
from qp4p_output import create_standardized_output
from qp4p_linear_system import add_linear_system_args, get_linear_system


# Matrix parsing moved to qp4p_linear_system helper


def inversion(qf, res=None):
    """
    Performs eigenvalue inversion λ -> λ^-1 using bit reversal.
    
    Works for eigenvalues of the form λ = 2^-k. Matrix A should be
    preprocessed to have eigenvalues as powers of 2.
    
    Args:
        qf: QuantumFloat input containing eigenvalue
        res: QuantumFloat result (created if None)
    
    Returns:
        QuantumFloat containing the inverted value (1/λ)
    """
    if res is None:
        res = QuantumFloat(qf.size + 1)

    # Bit reversal inversion (works for λ = 2^-k)
    for i in range(qf.size):
        cx(qf[i], res[qf.size - i])

    return res


def HHL_encoding(vector_b, hamiltonian_evolution, n, precision):
    """
    Performs HHL algorithm steps: prepare |b>, QPE, eigenvalue inversion.
    
    Args:
        vector_b: Input vector
        hamiltonian_evolution: Function performing e^(itA)
        n: Number of qubits encoding state |b>
        precision: Precision of quantum phase estimation
    
    Returns:
        Tuple of (qf, qpe_res, inv_res)
    """
    # Step 1: Prepare the state |b>
    qf = QuantumFloat(n)
    prepare(qf, vector_b, reversed=True)

    # Step 2: Apply quantum phase estimation
    qpe_res = QPE(qf, hamiltonian_evolution, precision=precision)

    # Step 3: Perform eigenvalue inversion
    inv_res = inversion(qpe_res)

    return qf, qpe_res, inv_res


def HHL(vector_b, hamiltonian_evolution, n, precision):
    """
    Main HHL algorithm function.
    
    Solves Ax=b by preparing quantum state |x> with amplitudes
    proportional to the solution x.
    
    Args:
        vector_b: Input vector
        hamiltonian_evolution: Function performing e^(itA)
        n: Number of qubits encoding state |b> (N = 2^n)
        precision: Precision of quantum phase estimation
    
    Returns:
        QuantumFloat containing the solution state |x>
    """
    # Perform encoding (Steps 1-3)
    qf, qpe_res, inv_res = HHL_encoding(vector_b, hamiltonian_evolution, n, precision)

    # Step 4: Uncompute auxiliary variables
    with invert():
        QPE(qf, hamiltonian_evolution, target=qpe_res)
        inversion(qpe_res, res=inv_res)

    # Reverse endianness for compatibility
    for i in range(qf.size // 2):
        swap(qf[i], qf[n - i - 1])

    return qf


def preprocess_matrix_for_hhl(matrix_A):
    """
    Preprocesses matrix A to ensure eigenvalues are powers of 2.
    
    Scales the matrix so eigenvalues become 2^-k for integer k,
    which is required for bit-reversal inversion.
    
    Args:
        matrix_A: Input Hermitian matrix
    
    Returns:
        Tuple of (A_scaled, scale_factor, eigenvalues_original, eigenvalues_scaled)
    """
    eigenvalues = np.linalg.eigvals(matrix_A)
    max_eigenval = np.max(np.abs(eigenvalues))
    
    # Scale so largest eigenvalue magnitude is close to 0.5 (2^-1)
    scale_factor = 0.5 / max_eigenval
    matrix_A_scaled = matrix_A * scale_factor
    eigenvalues_scaled = np.linalg.eigvals(matrix_A_scaled)

    return matrix_A_scaled, scale_factor, eigenvalues, eigenvalues_scaled


def create_hamiltonian_evolution(matrix_A, t=-np.pi, steps=1):
    """
    Creates a Hamiltonian evolution function from matrix A.
    
    Args:
        matrix_A: Hermitian matrix (should be preprocessed)
        t: Evolution time (default: -π)
        steps: Number of Trotter steps
    
    Returns:
        Function that applies e^(itA) to a quantum state
    """
    H = QubitOperator.from_matrix(matrix_A).to_pauli()

    def U(qf):
        H.trotterization()(qf, t=t, steps=steps)

    return U


def extract_solution_from_counts(counts, n_qubits_matrix):
    """
    Extracts the HHL solution vector from measurement counts.
    
    The solution is encoded in the last n_qubits_matrix bits.
    
    Args:
        counts: Dictionary of measurement counts (keys can be int, str, or tuple)
        n_qubits_matrix: Number of qubits encoding the matrix dimension
    
    Returns:
        Normalized quantum solution vector as numpy array
    """
    n_solution = 2 ** n_qubits_matrix
    total_shots = sum(counts.values())
    quantum_solution = np.zeros(n_solution)

    # Extract solution from the last n_qubits_matrix bits
    for key, count in counts.items():
        # Handle different key formats (Qrisp uses tuples, Qiskit uses strings)
        if isinstance(key, tuple):
            # Qrisp format: tuple of qubit values
            state_index = sum(bit * (2 ** i) for i, bit in enumerate(reversed(key)))
        elif isinstance(key, int):
            # Direct integer format
            state_index = key
        elif isinstance(key, str):
            # Qiskit format: bitstring
            if key.startswith('0x'):
                state_index = int(key, 16)
            else:
                state_index = int(key, 2)
        else:
            continue

        solution_bits = state_index & ((1 << n_qubits_matrix) - 1)

        if solution_bits < n_solution:
            # Use sqrt of probability to get amplitudes, not raw probabilities
            prob = count / total_shots
            quantum_solution[solution_bits] += np.sqrt(prob)

    # Filter near-zero components
    quantum_solution[np.abs(quantum_solution) < 1e-10] = 0

    # Normalize
    if np.linalg.norm(quantum_solution) > 0:
        quantum_solution = quantum_solution / np.linalg.norm(quantum_solution)

    return quantum_solution


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Ax=b using HHL algorithm with Qrisp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ax_equals_b_hhl_qrisp.py --size 2
  python ax_equals_b_hhl_qrisp.py --size 2 --seed 42
  python ax_equals_b_hhl_qrisp.py --size 2 --tridiag
  python ax_equals_b_hhl_qrisp.py --matrix '[[2,1],[1,2]]' --vector '[1,0]'
""")
    add_linear_system_args(parser)
    parser.add_argument("--precision", type=int, default=3,
                        help="QPE precision (number of bits, default: 3)")
    parser.add_argument("--trotter-steps", type=int, default=1,
                        help="Number of Trotter steps for Hamiltonian simulation (default: 1)")
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
    
    n = matrix_A.shape[0]

    # Preprocess matrix
    matrix_A_scaled, scale_factor, eigenvalues_orig, eigenvalues_scaled = \
        preprocess_matrix_for_hhl(matrix_A)

    # Create Hamiltonian evolution
    unitary = create_hamiltonian_evolution(matrix_A_scaled, t=-np.pi, steps=args.trotter_steps)

    # Run HHL (suppress Qrisp simulation output)
    n_qubits = int(np.log2(len(vector_b)))
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        with redirect_stdout(devnull):
            solution_qf = HHL(tuple(vector_b), unitary, n=n_qubits, precision=args.precision)

    # Convert Qrisp circuit to Qiskit
    quantum_session = solution_qf.qs()
    circuit = quantum_session.to_qiskit()
    
    # Deep decomposition to expand all custom gates to basis gates
    # Qrisp uses nested custom gates that need many decomposition passes
    circuit_decomposed = circuit
    for _ in range(10):  # Deep decomposition
        try:
            circuit_decomposed = circuit_decomposed.decompose()
        except Exception:
            break  # Stop if decomposition fails
    
    circuit_decomposed.measure_all()
    
    # Set up backend with noise model support
    noise_model, fake_backend, _ = build_noise_model(args.t1, args.t2, args.backend, args.coupling_map)
    
    if fake_backend:
        backend = Aer.get_backend('aer_simulator')
        backend = backend.from_backend(fake_backend)
    else:
        backend = Aer.get_backend('aer_simulator')
    
    # Transpile for execution
    transpiled = transpile(
        circuit_decomposed,
        backend=backend,
        optimization_level=3
    )
    
    # Execute on Aer
    if noise_model:
        job = backend.run(transpiled, shots=args.shots, noise_model=noise_model)
    else:
        job = backend.run(transpiled, shots=args.shots)
    
    result = job.result()
    counts = result.get_counts()

    # Extract solution
    quantum_solution = extract_solution_from_counts(counts, n_qubits)

    # Compute classical solution for comparison
    classical_solution = np.linalg.solve(matrix_A, vector_b)
    classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)

    # Keep raw quantum solution before normalization
    quantum_solution_raw = quantum_solution.copy()
    
    # Compute error metrics on normalized solutions
    l2_error = np.linalg.norm(quantum_solution - classical_solution_normalized)
    fidelity = np.abs(np.dot(quantum_solution, classical_solution_normalized))**2

    # Build standardized output
    output = create_standardized_output(
        algorithm="hhl_qrisp",
        script_name="ax_equals_b_hhl_qrisp.py",
        problem={
            "matrix": matrix_A.tolist(),
            "rhs": vector_b.tolist(),
            "dimension": n,
            "condition_number": float(sys_metadata['condition_number']),
            "eigenvalues_original": [float(e) for e in eigenvalues_orig],
            "eigenvalues_scaled": [float(e) for e in eigenvalues_scaled],
            "scale_factor": float(scale_factor)
        },
        config={
            "precision": args.precision,
            "trotter_steps": args.trotter_steps,
            "shots": args.shots,
            "backend": args.backend if args.backend else "aer_simulator",
            "coupling_map": args.coupling_map
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
        circuit_info={
            "num_qubits": transpiled.num_qubits,
            "depth": transpiled.depth(),
            "gate_counts": dict(transpiled.count_ops())
        }
    )

    print(json.dumps(output, indent=2))
