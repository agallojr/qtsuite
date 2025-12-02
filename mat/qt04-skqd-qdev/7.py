# Imports
from qiskit_ibm_catalog import QiskitServerless
from qc_grader.challenges.qdc_2025 import qdc25_lab4

import skqd_helpers


# Instantiate a `QsikitServerless` object
client = QiskitServerless(name='qdc-2025')


# Set Hamiltonian parameters
num_orbs = 24
hybridization = 1.0
hopping = 1.0
onsite = 1.0
chemical_potential = -0.5 * onsite

# We need to upload the program intended to run to the Serverless cloud environment.
# (And re-upload it any time we change it's source code!)

# `True` to run locally, `False` to run in Serverless environment
local = False  # Using serverless with quantum job ID

# Quantum job ID from IBM Quantum
job_id = 'd4b6vi7nmdfs73adjbs0'  # Boston job

# SQD Options
energy_tol = 1e-4
occupancies_tol = 1e-3
max_iterations = 10  # Feel free to modify

# Eigenstate solver options
num_batches = 5  # Feel free to modify
samples_per_batch = 200  # Feel free to modify
max_cycle = 200
symmetrize_spin = True
carryover_threshold = 1e-5

# Note: bit_array will be retrieved from job_id by serverless function

# Build SIAM Hamiltonian
hcore, eri = skqd_helpers.siam_hamiltonian(
    num_orbs, hopping, onsite, hybridization, chemical_potential
)

# Call `skqd_helpers.classically_diagonalize()` and store the results
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
    local=local
    )

qdc25_lab4.grade_lab4_ex6(result, num_orbs, hopping, onsite, hybridization, chemical_potential)
