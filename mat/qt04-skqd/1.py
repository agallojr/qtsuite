"""
ex1
"""

# Imports
from qiskit.quantum_info import SparsePauliOp
import numpy as np

from qc_grader.challenges.qdc_2025 import qdc25_lab4

import skqd_helpers

def perturbed_tfim_hamiltonian(num_qubits: int, J: float, hx: float, hz: float) -> SparsePauliOp:
    """Builds the perturbed transverse-field Ising model Hamiltonian as a `SparsePauliOp`.

    Args:
        num_qubits: Number of qubits
        J: Exchange energy
        hx: Field strength in x-direction
        hz: Field strength in z-direction

    Returns:
        Hamiltonian as a SparsePauliOp following Qiskit's endian convention.
    """
    pauli_list = []
    coeff_list = []

    # ZZ terms
    for j in range(num_qubits - 1):
        # Build Pauli string with Z at positions j and j+1
        pauli_str = ['I'] * num_qubits
        pauli_str[j] = 'Z'
        pauli_str[j + 1] = 'Z'
        pauli_list.append(''.join(pauli_str))
        coeff_list.append(-J)

    # X terms
    for j in range(num_qubits):
        # Build Pauli string with X at position j
        pauli_str = ['I'] * num_qubits
        pauli_str[j] = 'X'
        pauli_list.append(''.join(pauli_str))
        coeff_list.append(-hx)

    # Z term
    pauli_str = ['I'] * num_qubits
    pauli_str[0] = 'Z'
    pauli_list.append(''.join(pauli_str))
    coeff_list.append(-hz)

    return SparsePauliOp(pauli_list, coeffs=np.array(coeff_list, dtype=np.complex64))


if __name__ == "__main__":
    # Construct example Hamiltonian
    test_hamiltonian = perturbed_tfim_hamiltonian(num_qubits=6, J=0.1, hx=0.2, hz=0.3)
    paulis = test_hamiltonian.paulis
    coeffs = test_hamiltonian.coeffs

    # Expected Pauli terms
    expected_paulis = [
        'ZZIIII', 'IZZIII', 'IIZZII', 'IIIZZI',
        'IIIIZZ', 'XIIIII', 'IXIIII', 'IIXIII',
        'IIIXII', 'IIIIXI', 'IIIIIX', 'ZIIIII'
        ]
    # Expected Pauli coefficients
    expected_coeffs = np.array([
        -0.1+0.j, -0.1+0.j, -0.1+0.j, -0.1+0.j,
        -0.1+0.j, -0.2+0.j, -0.2+0.j, -0.2+0.j,
        -0.2+0.j, -0.2+0.j, -0.2+0.j, -0.30000001+0.j
        ])

    # Checks

    print("Pauli terms:", paulis)
    print("Expected Pauli terms:", expected_paulis)
    print("Pauli coefficients:", coeffs)
    print("Expected Pauli coefficients:", expected_coeffs)

    assert paulis == expected_paulis
    assert np.allclose(coeffs, expected_coeffs)

    qdc25_lab4.grade_lab4_ex1(perturbed_tfim_hamiltonian)
