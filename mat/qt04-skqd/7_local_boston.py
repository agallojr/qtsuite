# Imports
from qc_grader.challenges.qdc_2025 import qdc25_lab4
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
max_iterations = 3  # Reduced to finish faster

# Eigenstate solver options
num_batches = 3  # Reduced from 5
samples_per_batch = 100  # Reduced from 200 - key to prevent stalling
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

print("\nGrading result...")
qdc25_lab4.grade_lab4_ex6(result, num_orbs, hopping, onsite, hybridization, chemical_potential)
