"""
Step 4: Construct SIAM Hamiltonian in momentum basis.
"""

import numpy as np

import skqd_helpers


def siam_hamiltonian_momentum(
    num_orbs: int,
    hopping: float,
    onsite: float,
    hybridization: float,
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct SIAM Hamiltonian in momentum basis.
    
    Args:
        num_orbs: Number of spatial orbitals
        hopping: Hopping parameter
        onsite: Onsite energy
        hybridization: Hybridization strength
        chemical_potential: Chemical potential
        
    Returns:
        Tuple of (h1e, h2e) arrays in momentum basis.
    """
    h1e, h2e = skqd_helpers.siam_hamiltonian(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    
    orbital_rotation = skqd_helpers.momentum_basis(num_orbs)
    
    # Transpose orbital_rotation to get proper U.H @ H @ U transformation
    h1e_momentum, h2e_momentum = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)
    
    return h1e_momentum, h2e_momentum


def run_step1():
    """Run step 4: construct SIAM Hamiltonian in momentum basis."""
    NUM_ORBS = 10
    HOPPING = 1.0
    ONSITE = 5
    HYBRIDIZATION = 1.0
    CHEMICAL_POTENTIAL = -0.5 * ONSITE

    result = siam_hamiltonian_momentum(NUM_ORBS, HYBRIDIZATION, HOPPING, ONSITE, CHEMICAL_POTENTIAL)
    
    print(f"h1e shape: {result[0].shape}")
    print(f"h2e shape: {result[1].shape}")
    print("Step 4 passed: SIAM Hamiltonian constructed in momentum basis.")
    
    return result


if __name__ == "__main__":
    run_step1()
