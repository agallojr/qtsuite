import numpy as np
import skqd_helpers

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE

print("Grader calls with:")
print(f"  NUM_ORBS={NUM_ORBS}, HYBRIDIZATION={HYBRIDIZATION}, HOPPING={HOPPING}, ONSITE={ONSITE}, CHEMICAL_POTENTIAL={CHEMICAL_POTENTIAL}")
print()

# What the function should do when called by grader
def siam_hamiltonian_momentum_grader_order(
    num_orbs: int,
    hybridization: float,
    hopping: float,
    onsite: float,
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match grader's parameter order"""
    print(f"Function receives:")
    print(f"  num_orbs={num_orbs}, hybridization={hybridization}, hopping={hopping}, onsite={onsite}, chemical_potential={chemical_potential}")
    
    h1e, h2e = skqd_helpers.siam_hamiltonian(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    
    orbital_rotation = skqd_helpers.momentum_basis(num_orbs)
    
    h1e_momentum, h2e_momentum = skqd_helpers.rotated(h1e, h2e, orbital_rotation)
    
    return h1e_momentum, h2e_momentum

print("Testing with grader's call pattern:")
h1e, h2e = siam_hamiltonian_momentum_grader_order(
    NUM_ORBS, HYBRIDIZATION, HOPPING, ONSITE, CHEMICAL_POTENTIAL
)

print()
print(f"h1e shape: {h1e.shape}")
print(f"h1e[0:3, 0:3]:")
print(h1e[0:3, 0:3])
print()
print(f"h1e diagonal:")
print(np.diag(h1e))
