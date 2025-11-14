# Imports
from qiskit.primitives import BitArray
import numpy as np
import skqd_helpers


# Set Hamiltonian parameters
num_orbs = 24
hybridization = 1.0
hopping = 1.0
onsite = 1.0
chemical_potential = -0.5 * onsite

# Local execution
local = True

# SQD Options
energy_tol = 1e-4
occupancies_tol = 1e-3
max_iterations = 6  # Fast convergence

# Eigenstate solver options
num_batches = 5  # Reduced for speed
samples_per_batch = 100  # Fast but still effective
max_cycle = 200
symmetrize_spin = True
carryover_threshold = 1e-5

# Load Boston bit array data
print("Loading bit_array_boston.npz...")
data = np.load('bit_array_boston.npz')
bit_array = BitArray(data['samples'], num_bits=int(data['num_bits']))
print(f"Loaded bit array: shape={bit_array.array.shape}, num_bits={bit_array.num_bits}")

# Build SIAM Hamiltonian
print("Building SIAM Hamiltonian...")
hcore, eri = skqd_helpers.siam_hamiltonian(
    num_orbs, hopping, onsite, hybridization, chemical_potential
)

# Call `skqd_helpers.classically_diagonalize()` and store the results
print("Starting local diagonalization...")
print("="*60)
result = skqd_helpers.classically_diagonalize(
    bit_array=bit_array,
    hcore=hcore,
    eri=eri,
    num_orbitals=num_orbs,
    nelec=num_orbs,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=max_iterations,
    num_batches=num_batches,
    samples_per_batch=samples_per_batch,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    max_cycle=max_cycle,
    local=local
)

print("="*60)
print("COMPUTATION COMPLETE")
print("="*60)
print(f"Energy history: {result}")
print(f"Final energy: {result[-1]:.6f}")
print(f"Number of iterations: {len(result)}")
print("="*60)

# Save result
np.savez('result_boston.npz', 
         energy_history=result,
         num_orbs=num_orbs,
         hopping=hopping,
         onsite=onsite,
         hybridization=hybridization,
         chemical_potential=chemical_potential)
print("\nResult saved to result_boston.npz")
print("Run submit_boston.py to grade this result")
