"""
HHL (Harrow-Hassidim-Lloyd) Algorithm Implementation using Qrisp

This implementation constructs the HHL quantum circuit for solving linear systems Ax=b.
Based on: https://qrisp.eu/general/tutorial/HHL.html
"""

#pylint: disable=invalid-name

import numpy as np
from qrisp import QuantumFloat, QPE, prepare, cx, swap, invert
from qrisp.operators import QubitOperator
from qiskit_aer import Aer
from qiskit import transpile


def inversion(qf, res=None):
    """
    Performs inversion λ -> λ^-1 for arbitrary eigenvalues.
    
    Uses bit reversal which works for eigenvalues of the form λ = 2^-k.
    For truly arbitrary eigenvalues, Qrisp would need a general division implementation.
    
    Note: This means your input matrix A should have eigenvalues that are powers of 2,
    or you need to scale/transform A appropriately.
    
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
    Performs Steps 1-4 of the HHL algorithm (simplified without RUS).
    
    Steps:
    1. Prepare state |b>
    2. Apply quantum phase estimation to get eigenvalues
    3. Perform eigenvalue inversion
    
    Args:
        b: Input vector
        hamiltonian_evolution: Function performing e^(itA)
        n: Number of qubits encoding state |b>
        precision: Precision of quantum phase estimation
    
    Returns:
        Tuple of (qf, qpe_res, inv_res)
    """
    # Step 1: Prepare the state |b>
    qf = QuantumFloat(n)
    # Reverse endianness for compatibility with Hamiltonian simulation
    prepare(qf, vector_b, reversed=True)

    # Step 2: Apply quantum phase estimation
    qpe_res = QPE(qf, hamiltonian_evolution, precision=precision)

    # Step 3: Perform eigenvalue inversion
    inv_res = inversion(qpe_res)

    return qf, qpe_res, inv_res


def HHL(vector_b, hamiltonian_evolution, n, precision):
    """
    Main HHL algorithm function.
    
    Solves the linear system Ax=b by preparing quantum state |x> with amplitudes
    proportional to the solution x.
    
    Args:
        b: Input vector
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

    # Reverse endianness for compatibility with Hamiltonian simulation
    for i in range(qf.size // 2):
        swap(qf[i], qf[n - i - 1])

    return qf


def preprocess_matrix_for_hhl(matrix_A):
    """
    Preprocesses matrix A to ensure eigenvalues are powers of 2.
    
    This is required for the bit-reversal inversion method to work correctly.
    The function scales the matrix so that eigenvalues become 2^-k for integer k.
    
    Args:
        A: Input Hermitian matrix
        target_precision: Number of bits for QPE precision (determines eigenvalue resolution)
    
    Returns:
        Tuple of (A_scaled, scale_factor, eigenvalues_original, eigenvalues_scaled)
    """
    # Get eigenvalues of original matrix
    eigenvalues = np.linalg.eigvals(matrix_A)

    # Find the maximum absolute eigenvalue
    max_eigenval = np.max(np.abs(eigenvalues))

    # Scale so largest eigenvalue magnitude is close to 0.5 (2^-1)
    # This ensures eigenvalues fit in the range that QPE can represent
    scale_factor = 0.5 / max_eigenval
    matrix_A_scaled = matrix_A * scale_factor

    # Verify scaled eigenvalues
    eigenvalues_scaled = np.linalg.eigvals(matrix_A_scaled)

    print(f"Original eigenvalues: {eigenvalues}")
    print(f"Scale factor: {scale_factor}")
    print(f"Scaled eigenvalues: {eigenvalues_scaled}")
    print(f"Max scaled eigenvalue: {np.max(np.abs(eigenvalues_scaled))}")

    return matrix_A_scaled, scale_factor, eigenvalues, eigenvalues_scaled


def create_hamiltonian_evolution(matrix_A, t=-np.pi, steps=1):
    """
    Creates a Hamiltonian evolution function from matrix A.
    
    Args:
        matrix_A: Hermitian matrix (should be preprocessed with preprocess_matrix_for_hhl)
        t: Evolution time (default: -π)
        steps: Number of Trotter steps
    
    Returns:
        Function that applies e^(itA) to a quantum state
    """
    H = QubitOperator.from_matrix(matrix_A).to_pauli()

    def U(qf):
        H.trotterization()(qf, t=t, steps=steps)

    return U


def get_hhl_circuit(matrix_A, vector_b, precision=3, steps=1):
    """
    High-level function to solve Ax=b using HHL algorithm.
    
    This function handles all preprocessing and returns the Qiskit circuit.
    
    Args:
        A: Input Hermitian matrix (numpy array)
        b: Input vector (numpy array)
        precision: QPE precision (number of bits)
        steps: Number of Trotter steps for Hamiltonian simulation
        verbose: Print preprocessing information
    
    Returns:
        qiskit_circuit: Qiskit QuantumCircuit ready to execute
    """
    # Preprocess matrix
    matrix_A_scaled, _, _, _ = preprocess_matrix_for_hhl(matrix_A)

    # Create Hamiltonian evolution
    unitary = create_hamiltonian_evolution(matrix_A_scaled, t=-np.pi, steps=steps)

    # Run HHL
    n_qubits = int(np.log2(len(vector_b)))
    solution_x = HHL(tuple(vector_b), unitary, n=n_qubits, precision=precision)

    # Convert to Qiskit
    circuit = convert_to_qiskit(solution_x)
    circuit.measure_all()
    return circuit


def convert_to_qiskit(qrisp_variable):
    """
    Converts a Qrisp QuantumVariable to a Qiskit QuantumCircuit.
    
    In Qrisp, when you create quantum variables and apply gates, those operations
    are recorded in a QuantumSession (the circuit). Each QuantumVariable has a
    reference to its QuantumSession via the .qs() method.
    
    This function extracts that QuantumSession and converts it to Qiskit format.
    
    Args:
        qrisp_variable: Qrisp QuantumVariable (like QuantumFloat)
                       The variable contains the circuit that created it
    
    Returns:
        Qiskit QuantumCircuit containing all gates applied to create the variable
    """
    # Get the QuantumSession (circuit) from the QuantumVariable
    if hasattr(qrisp_variable, 'qs'):
        quantum_session = qrisp_variable.qs()
    else:
        # If it's already a QuantumSession, use it directly
        quantum_session = qrisp_variable

    # Convert the QuantumSession to Qiskit format
    circuit = quantum_session.to_qiskit()

    return circuit


def extract_solution_from_counts(counts, n_qubits_matrix):
    """
    Extracts the HHL solution vector from measurement counts.
    
    The HHL circuit encodes the solution in the last n_qubits_matrix bits
    of the measurement results. This function extracts those bits and
    constructs the normalized solution vector.
    
    Args:
        counts: Dictionary of measurement counts from Qiskit result
        n_qubits_matrix: Number of qubits encoding the matrix dimension
    
    Returns:
        Normalized quantum solution vector as numpy array
    """
    n_solution = 2 ** n_qubits_matrix
    total_shots = sum(counts.values())
    quantum_solution = np.zeros(n_solution)

    # Extract solution from the last n_qubits_matrix bits (solution register)
    for bitstring, count in counts.items():
        # Convert bitstring to state index (handle hex and binary formats)
        if bitstring.startswith('0x'):
            state_index = int(bitstring, 16)
        else:
            state_index = int(bitstring, 2)

        # Extract solution register bits (last n_qubits_matrix bits)
        solution_bits = state_index & ((1 << n_qubits_matrix) - 1)

        if solution_bits < n_solution:
            quantum_solution[solution_bits] += count / total_shots

    # Filter near-zero components
    quantum_solution[np.abs(quantum_solution) < 1e-10] = 0

    # Normalize
    if np.linalg.norm(quantum_solution) > 0:
        quantum_solution = quantum_solution / np.linalg.norm(quantum_solution)

    return quantum_solution


def _solve_hhl_internal(matrix_A, vector_b, precision=3, steps=1, shots=1024, verbose=True):
    """
    Internal implementation of HHL solver with full control over parameters.
    
    Args:
        matrix_A: Hermitian matrix (numpy array)
        vector_b: Input vector (numpy array)
        precision: QPE precision (number of bits)
        steps: Number of Trotter steps for Hamiltonian simulation
        shots: Number of measurement shots
        verbose: Print circuit statistics and error metrics
    
    Returns:
        Normalized quantum solution vector as numpy array
    """
    # Get HHL circuit
    qiskit_circuit = get_hhl_circuit(matrix_A, vector_b, precision=precision, steps=steps)

    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(
        qiskit_circuit,
        simulator,
        optimization_level=3,
        basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset']
    )

    if verbose:
        print(f"Transpiled circuit qubits: {transpiled.num_qubits}")
        print(f"Transpiled circuit depth: {transpiled.depth()}")
        print(f"Transpiled gate counts: {transpiled.count_ops()}")

    # Run on simulator
    job = simulator.run(transpiled, shots=shots)  # type: ignore
    result = job.result()  # type: ignore
    counts = result.get_counts()  # type: ignore

    # Extract solution from measurement counts
    n_qubits_matrix = int(np.log2(len(vector_b)))
    quantum_solution = extract_solution_from_counts(counts, n_qubits_matrix)

    if verbose:
        print(f"\nQuantum solution (normalized): {quantum_solution}")

        # Compare with classical solution
        classical_solution = np.linalg.solve(matrix_A, vector_b)
        classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)

        print(f"Classical solution (normalized): {classical_solution_normalized}")

        # Compute error metrics
        l2_error = np.linalg.norm(quantum_solution - classical_solution_normalized)
        fidelity = np.abs(np.dot(quantum_solution, classical_solution_normalized))**2

        print("\nError Metrics:")
        print(f"  L2 norm difference: {l2_error:.6f}")
        print(f"  Fidelity: {fidelity:.6f}")

    return quantum_solution


def solve_hhl(matrix_A, vector_b):
    """
    Solves the linear system Ax=b using the HHL algorithm.
    
    Simple interface with sensible defaults. For advanced usage, see _solve_hhl_internal.
    
    Args:
        matrix_A: Hermitian matrix (numpy array)
        vector_b: Input vector (numpy array)
    
    Returns:
        Normalized quantum solution vector as numpy array
    """
    return _solve_hhl_internal(
        matrix_A,
        vector_b,
        precision=3,
        steps=1,
        shots=1024,
        verbose=True
    )


if __name__ == "__main__":
    # Example: arbitrary matrix and vector
    A = np.array([[3/8, 1/8], [1/8, 3/8]])
    b = np.array([1, 1])

    x = solve_hhl(A, b)
    print(f"\n*** Solution: {x}\n")
