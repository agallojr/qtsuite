"""Calculate the exact ground state energy for comparison"""

import numpy as np
import skqd_helpers

# Load the parameters used
data = np.load('result_boston.npz')
num_orbs = int(data['num_orbs'])
hopping = float(data['hopping'])
onsite = float(data['onsite'])
hybridization = float(data['hybridization'])
chemical_potential = float(data['chemical_potential'])
boston_energy = data['energy_history'][-1]

print("="*70)
print("EXACT GROUND STATE ENERGY CALCULATION")
print("="*70)

print("\nParameters:")
print(f"  num_orbs = {num_orbs}")
print(f"  hopping = {hopping}")
print(f"  onsite = {onsite}")
print(f"  hybridization = {hybridization}")
print(f"  chemical_potential = {chemical_potential}")

# Build the Hamiltonian
print("\nBuilding SIAM Hamiltonian...")
hcore, eri = skqd_helpers.siam_hamiltonian(
    num_orbs, hopping, onsite, hybridization, chemical_potential
)

# Calculate exact ground state using Full Configuration Interaction (FCI)
print("Performing exact diagonalization (FCI)...")
from pyscf import fci

# Setup FCI solver
nelec = num_orbs  # Half-filled system
norb = num_orbs

# Solve for ground state using direct FCI
exact_energy, _ = fci.direct_spin1.kernel(hcore, eri, norb, (nelec//2, nelec//2))

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"\nExact Ground State Energy (FCI): {exact_energy:.8f} Ha")
print(f"Boston SQD Result:                {boston_energy:.8f} Ha")
print(f"\nError: {abs(boston_energy - exact_energy):.8f} Ha")
print(f"Relative Error: {abs(boston_energy - exact_energy)/abs(exact_energy)*100:.4f}%")

# Check against common tolerance levels
print("\n" + "="*70)
print("TOLERANCE CHECK")
print("="*70)
tolerances = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
for tol in tolerances:
    status = "✓ PASS" if abs(boston_energy - exact_energy) < tol else "✗ FAIL"
    print(f"  {status}  Tolerance: {tol:.0e} Ha")

print("="*70)
