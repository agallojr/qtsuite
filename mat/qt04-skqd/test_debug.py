import numpy as np
import skqd_helpers

NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE

print("Testing siam_hamiltonian_momentum")
print("=" * 50)

# Get position basis Hamiltonian
h1e_pos, h2e_pos = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

print(f"h1e_pos shape: {h1e_pos.shape}")
print(f"h1e_pos[0:3, 0:3]:")
print(h1e_pos[0:3, 0:3])
print()

# Get orbital rotation
orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)
print(f"orbital_rotation shape: {orbital_rotation.shape}")
print(f"Is unitary: {np.allclose(orbital_rotation @ orbital_rotation.T, np.eye(NUM_ORBS))}")
print()

# Rotate to momentum basis
h1e_mom, h2e_mom = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation)

print(f"h1e_mom shape: {h1e_mom.shape}")
print(f"h1e_mom is Hermitian: {np.allclose(h1e_mom, h1e_mom.conj().T)}")
print(f"h1e_mom[0:3, 0:3]:")
print(h1e_mom[0:3, 0:3])
print()

# Check if diagonal dominance (should be sparse in momentum basis)
print("Diagonal elements:")
print(np.diag(h1e_mom))
print()
print("Max off-diagonal element:")
h1e_mom_nodiag = h1e_mom.copy()
np.fill_diagonal(h1e_mom_nodiag, 0)
print(np.max(np.abs(h1e_mom_nodiag)))
