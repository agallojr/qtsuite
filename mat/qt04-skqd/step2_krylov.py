"""
Step 2: Construct Krylov circuits for SIAM.
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
            for instruction in \
                skqd_helpers.trotter_step(qreg, dt, hamiltonian_unitary,
                    impurity_index, num_orbs):
                qc.append(instruction)
        
        circuits.append(qc)
    
    return circuits


def run_step2(
    num_orbs: int = 10,
    hopping: float = 1.0,
    onsite: float = 5.0,
    hybridization: float = 1.0,
    krylov_dim: int = 5,
    filling_factor: float = -0.5,
    dt_multiplier: float = 1.0,
    add_measurements: bool = True,
) -> list[QuantumCircuit]:
    """Run step 2: construct SIAM Krylov circuits.
    
    Args:
        num_orbs: Number of spatial orbitals (qubits = 2 * num_orbs)
        hopping: Hopping parameter
        onsite: Onsite energy (U)
        hybridization: Hybridization strength
        krylov_dim: Number of Krylov basis states
        filling_factor: Multiplier for chemical potential
        dt_multiplier: Multiplier for time step (default 1.0, try 6-10 for more evolution)
        add_measurements: Whether to add measurement gates
        
    Returns:
        List of Krylov circuits.
    """
    chemical_potential = filling_factor * onsite
    
    hamiltonian = siam_hamiltonian_momentum(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    
    dt = dt_multiplier * np.pi / np.linalg.norm(hamiltonian[0], ord=2)
    impurity_index = (num_orbs - 1) // 2
    
    circuits = construct_krylov_siam(
        num_orbs, impurity_index, hamiltonian, dt, krylov_dim
    )
    
    if add_measurements:
        for qc in circuits:
            qc.measure_all()
    
    num_qubits = 2 * num_orbs
    print(f"Constructed {len(circuits)} SIAM Krylov circuits ({num_qubits} qubits).")
    
    return circuits

