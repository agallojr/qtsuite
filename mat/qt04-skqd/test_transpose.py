import numpy as np
import skqd_helpers

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE

# Get position basis Hamiltonian
h1e_pos, h2e_pos = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)

print("Test 1: Using orbital_rotation as-is")
h1e_mom1, h2e_mom1 = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation)
print(f"h1e_mom diagonal: {np.diag(h1e_mom1)}")
print()

print("Test 2: Using orbital_rotation.T")
h1e_mom2, h2e_mom2 = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation.T)
print(f"h1e_mom diagonal: {np.diag(h1e_mom2)}")
print()

print("Test 3: Manual rotation U.H @ h1e @ U")
h1e_mom3 = orbital_rotation.conj().T @ h1e_pos @ orbital_rotation
print(f"h1e_mom diagonal: {np.diag(h1e_mom3)}")
print()

print("Test 4: Manual rotation U @ h1e @ U.H")
h1e_mom4 = orbital_rotation @ h1e_pos @ orbital_rotation.conj().T
print(f"h1e_mom diagonal: {np.diag(h1e_mom4)}")
print()

print("Are Test 1 and Test 3 the same?", np.allclose(h1e_mom1, h1e_mom3))
print("Are Test 1 and Test 4 the same?", np.allclose(h1e_mom1, h1e_mom4))
