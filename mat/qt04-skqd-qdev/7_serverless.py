# Imports
from qiskit_ibm_catalog import QiskitServerless
import skqd_helpers

# Instantiate a QiskitServerless object
client = QiskitServerless(name='qdc-2025')

# Set Hamiltonian parameters
num_orbs = 24
hybridization = 1.0
hopping = 1.0
onsite = 1.0
chemical_potential = -0.5 * onsite

# Quantum job ID from IBM Quantum
job_id = 'd4b6vi7nmdfs73adjbs0'  # Boston job

# SQD Options - aggressive for deep convergence
energy_tol = 1e-4
occupancies_tol = 1e-3
max_iterations = 10  # Deep convergence to reach < -8.0

# Eigenstate solver options
num_batches = 8  # Better sampling
samples_per_batch = 150  # Safe from stalling
max_cycle = 200
symmetrize_spin = True
carryover_threshold = 1e-5

# Build SIAM Hamiltonian
print("Building SIAM Hamiltonian...")
hcore, eri = skqd_helpers.siam_hamiltonian(
    num_orbs, hopping, onsite, hybridization, chemical_potential
)

# Call serverless diagonalization
print("Submitting to serverless...")
print(f"Parameters: max_iter={max_iterations}, batches={num_batches}, samples={samples_per_batch}")
result = skqd_helpers.classically_diagonalize(
    hcore=hcore,
    eri=eri,
    num_orbitals=num_orbs,
    nelec=num_orbs,
    num_elec_a=num_orbs // 2,
    num_elec_b=num_orbs // 2,
    job_id=job_id,
    client=client,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=max_iterations,
    num_batches=num_batches,
    samples_per_batch=samples_per_batch,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    max_cycle=max_cycle,
    local=False
)

print("\n" + "="*60)
print("SERVERLESS RESULT:")
print("="*60)
print(f"Energy history: {result}")
print(f"Final energy: {result[-1]:.6f}")
print("="*60)

# Save result
import numpy as np
np.savez('result_serverless.npz', 
         energy_history=result,
         num_orbs=num_orbs,
         hopping=hopping,
         onsite=onsite,
         hybridization=hybridization,
         chemical_potential=chemical_potential)
print("\nResult saved to result_serverless.npz")
print("Run submit_serverless.py to grade")
