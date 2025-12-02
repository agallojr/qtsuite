"""
Step 6: Construct large SIAM Krylov circuits (24 orbitals, 48 qubits).

This is the production-scale problem that requires real hardware.
Circuit construction only - transpilation and execution are separate modules.
"""

import numpy as np

from qiskit import QuantumCircuit

from step1_siam import siam_hamiltonian_momentum
from step2_siam_small import construct_krylov_siam


def run_step2() -> list[QuantumCircuit]:
    """Run step 6: construct large SIAM Krylov circuits (48 qubits).
    
    Returns:
        List of SIAM Krylov circuits with measurements.
    """
    # Set Hamiltonian parameters (production scale)
    num_orbs = 24
    hybridization = 1.0
    hopping = 1.0
    onsite = 1.0
    chemical_potential = -0.5 * onsite

    # Construct SIAM Hamiltonian
    hamiltonian = siam_hamiltonian_momentum(num_orbs, hopping, onsite, hybridization,
        chemical_potential)

    # Calculate dt based on spectral norm
    dt = np.pi / np.linalg.norm(hamiltonian[0], ord=2)
    impurity_index = (num_orbs - 1) // 2  # Impurity is at center in momentum basis
    krylov_dim = 10

    # Construct Krylov circuits
    krylov_circuits = construct_krylov_siam(num_orbs, impurity_index, hamiltonian,
        dt, krylov_dim)

    # Add measurements to all circuits
    for qc in krylov_circuits:
        qc.measure_all()

    print(f"Constructed {len(krylov_circuits)} SIAM Krylov circuits (48 qubits).")
    print("Step 6 passed: Large SIAM circuits constructed.")
    
    return krylov_circuits


if __name__ == "__main__":
    run_step2()
