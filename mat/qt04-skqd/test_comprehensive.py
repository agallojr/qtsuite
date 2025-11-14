import numpy as np
import skqd_helpers

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE

print("="*60)
print("Testing all variations")
print("="*60)

# Get position basis Hamiltonian
h1e_pos, h2e_pos = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

print("\nPosition basis h1e:")
print(h1e_pos)
print("\nPosition basis h1e diagonal:", np.diag(h1e_pos))

# Get orbital rotation
orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)

print("\n" + "="*60)
print("Option 1: orbital_rotation (as-is)")
print("="*60)
h1e_mom, h2e_mom = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation)
print("h1e shape:", h1e_mom.shape)
print("h1e[0:3, 0:3]:")
print(h1e_mom[0:3, 0:3])
print("Diagonal:", np.diag(h1e_mom))

print("\n" + "="*60)
print("Option 2: orbital_rotation.T")
print("="*60)
h1e_mom2, h2e_mom2 = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation.T)
print("h1e shape:", h1e_mom2.shape)
print("h1e[0:3, 0:3]:")
print(h1e_mom2[0:3, 0:3])
print("Diagonal:", np.diag(h1e_mom2))

# Check which one is more diagonal (more sparse)
h1e_offdiag = h1e_mom.copy()
np.fill_diagonal(h1e_offdiag, 0)
norm_offdiag_1 = np.linalg.norm(h1e_offdiag, 'fro')

h1e_offdiag2 = h1e_mom2.copy()
np.fill_diagonal(h1e_offdiag2, 0)
norm_offdiag_2 = np.linalg.norm(h1e_offdiag2, 'fro')

print("\n" + "="*60)
print("Sparsity analysis (smaller = more diagonal/sparse)")
print("="*60)
print(f"Option 1 off-diagonal Frobenius norm: {norm_offdiag_1:.6f}")
print(f"Option 2 off-diagonal Frobenius norm: {norm_offdiag_2:.6f}")

if norm_offdiag_2 < norm_offdiag_1:
    print("\nOption 2 is MORE sparse (likely correct)")
else:
    print("\nOption 1 is MORE sparse (likely correct)")

# Also test data types
print("\n" + "="*60)
print("Data types:")
print("="*60)
print(f"h1e_mom dtype: {h1e_mom.dtype}")
print(f"h2e_mom dtype: {h2e_mom.dtype}")
