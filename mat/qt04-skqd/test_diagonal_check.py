import numpy as np
import skqd_helpers

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE

# Get the Hamiltonian
h1e, h2e = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)

print("WITH .T:")
h1e_mom_t, h2e_mom_t = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)
off_diag_t = h1e_mom_t.copy()
np.fill_diagonal(off_diag_t, 0)
print(f"  Max off-diagonal: {np.max(np.abs(off_diag_t)):.6f}")
print(f"  Off-diagonal norm: {np.linalg.norm(off_diag_t, 'fro'):.6f}")
print(f"  Diagonal: {np.diag(h1e_mom_t)}")

print("\nWITHOUT .T:")
h1e_mom, h2e_mom = skqd_helpers.rotated(h1e, h2e, orbital_rotation)
off_diag = h1e_mom.copy()
np.fill_diagonal(off_diag, 0)
print(f"  Max off-diagonal: {np.max(np.abs(off_diag)):.6f}")
print(f"  Off-diagonal norm: {np.linalg.norm(off_diag, 'fro'):.6f}")
print(f"  Diagonal: {np.diag(h1e_mom)}")

# Check if either is close to diagonal (off-diagonal elements near zero)
print(f"\nWITH .T is nearly diagonal: {np.max(np.abs(off_diag_t)) < 1e-10}")
print(f"WITHOUT .T is nearly diagonal: {np.max(np.abs(off_diag)) < 1e-10}")
