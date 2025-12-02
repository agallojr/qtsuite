"""
Step 5: Construct Krylov circuits for SIAM.
"""

import numpy as np
import scipy

from qiskit import QuantumCircuit, QuantumRegister

import skqd_helpers
from step1_siam import siam_hamiltonian_momentum


def construct_krylov_siam(
    num_orbs: int,
    impurity_index: int,
    hamiltonian: tuple[np.ndarray, np.ndarray],
    dt: float,
    krylov_dim: int
) -> list[QuantumCircuit]:
    """Generate Krylov circuits for SIAM.

    Args:
        num_orbs: Number of spatial orbitals
        impurity_index: Index of impurity orbital
        hamiltonian: one- and two-body Hamiltonian terms
        dt: Time step
        krylov_dim: Number of Krylov basis states
    
    Returns:
        SIAM Krylov circuits.
    """
    circuits = []
    
    # SIAM requires 2*num_orbs qubits (spin up and spin down)
    num_qubits = 2 * num_orbs
    
    # Occupancy number (half-filled system)
    occ_num = num_orbs // 2
    
    # Convert h1e to unitary time evolution operator for OrbitalRotationJW
    h1e_unitary = scipy.linalg.expm(-1j * dt * hamiltonian[0])
    hamiltonian_unitary = (h1e_unitary, hamiltonian[1])
    
    for k in range(krylov_dim):
        # Create quantum circuit with quantum register
        qreg = QuantumRegister(num_qubits, 'q')
        qc = QuantumCircuit(qreg)
        
        # Prepare initial state using helper
        for instruction in skqd_helpers.prepare_initial_state(qreg, num_orbs, occ_num):
            qc.append(instruction)
        
        # Apply k Trotter steps for k-th Krylov state using helper
        for _ in range(k):
            for instruction in skqd_helpers.trotter_step(qreg, dt, hamiltonian_unitary, impurity_index, num_orbs):
                qc.append(instruction)
        
        circuits.append(qc)
    
    return circuits


def run_step2():
    """Run step 5: construct SIAM Krylov circuits."""
    NUM_ORBS = 10
    HOPPING = 1.0
    ONSITE = 5
    HYBRIDIZATION = 1.0
    CHEMICAL_POTENTIAL = -0.5 * ONSITE
    
    hamiltonian = siam_hamiltonian_momentum(NUM_ORBS, HYBRIDIZATION, HOPPING, ONSITE,
        CHEMICAL_POTENTIAL)
    
    dt = np.pi / np.linalg.norm(hamiltonian[0], ord=2)
    impurity_index = (NUM_ORBS - 1) // 2
    krylov_dim = 5
    
    circuits = construct_krylov_siam(NUM_ORBS, impurity_index, hamiltonian, dt, krylov_dim)
    
    print(f"Constructed {len(circuits)} SIAM Krylov circuits.")
    print("Step 5 passed: SIAM Krylov circuits constructed.")
    
    return circuits


if __name__ == "__main__":
    run_step2()
