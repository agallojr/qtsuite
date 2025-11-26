"""Estimate ground state energy using lighter computational methods"""

import numpy as np
import skqd_helpers
from pyscf import scf, cc

# Load the parameters used
data = np.load('result_boston.npz')
num_orbs = int(data['num_orbs'])
hopping = float(data['hopping'])
onsite = float(data['onsite'])
hybridization = float(data['hybridization'])
chemical_potential = float(data['chemical_potential'])
boston_energy = data['energy_history'][-1]

print("="*70)
print("ENERGY ESTIMATION (Fast Methods)")
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

nelec = num_orbs
norb = num_orbs

print("\n" + "="*70)
print("METHOD 1: Hartree-Fock (HF) - Upper Bound")
print("="*70)
print("Computing HF energy (fast)...")

# Create a fake molecule object for PySCF
from pyscf import gto
mol = gto.M()
mol.nelectron = nelec
mol.spin = 0  # Singlet
mol.verbose = 0

# Run Hartree-Fock
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: hcore
mf.get_ovlp = lambda *args: np.eye(norb)
mf._eri = eri
mf.kernel()
hf_energy = mf.e_tot

print(f"HF Energy: {hf_energy:.6f} Ha")
print("(Note: HF provides an upper bound; exact energy should be lower)")

print("\n" + "="*70)
print("METHOD 2: CCSD - Better Approximation")
print("="*70)
print("Computing CCSD energy (moderate cost, ~30 sec - 2 min)...")

try:
    mycc = cc.CCSD(mf)
    mycc.kernel()
    ccsd_energy = mycc.e_tot
    print(f"CCSD Energy: {ccsd_energy:.6f} Ha")
    print("(Note: CCSD is typically within 1-2% of FCI for this system size)")
except Exception as e:
    print(f"CCSD calculation failed: {e}")
    ccsd_energy = None

print("\n" + "="*70)
print("COMPARISON WITH BOSTON RESULT")
print("="*70)
print(f"\nBoston SQD Result:     {boston_energy:.6f} Ha")
print(f"HF (upper bound):      {hf_energy:.6f} Ha")
if ccsd_energy:
    print(f"CCSD (approximation):  {ccsd_energy:.6f} Ha")
    print(f"\nBoston vs CCSD error:  {abs(boston_energy - ccsd_energy):.6f} Ha ({abs(boston_energy - ccsd_energy)*1000:.3f} mHa)")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
if boston_energy > hf_energy:
    print("⚠ WARNING: Boston result is HIGHER than HF upper bound!")
    print("   This suggests the SQD calculation has issues.")
else:
    print("✓ Boston result is below HF upper bound (good)")
    
if ccsd_energy and abs(boston_energy - ccsd_energy) < 0.001:
    print("✓ Boston result is very close to CCSD (< 1 mHa)")
elif ccsd_energy and abs(boston_energy - ccsd_energy) < 0.01:
    print("○ Boston result is reasonably close to CCSD (< 10 mHa)")
elif ccsd_energy:
    print(f"✗ Boston result differs from CCSD by {abs(boston_energy - ccsd_energy)*1000:.1f} mHa")
    print("  Likely needs more iterations or better sampling")

print("\nEstimate: True FCI energy likely in range:")
if ccsd_energy:
    estimated_fci = ccsd_energy - 0.05  # CCSD typically 1-2% high
    print(f"  [{estimated_fci:.6f}, {ccsd_energy:.6f}] Ha")
    print(f"\nIf grading tolerance is 1 mHa (0.001 Ha):")
    if abs(boston_energy - ccsd_energy) < 0.001:
        print("  ✓ Boston result likely PASSES")
    else:
        print("  ✗ Boston result likely FAILS")
        print(f"  Need to improve by ~{abs(boston_energy - ccsd_energy)*1000:.2f} mHa")
        
print("="*70)
