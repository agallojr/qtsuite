# Imports
import numpy as np
from qiskit.primitives import BitArray
from qiskit_ibm_runtime import QiskitRuntimeService

# Set Hamiltonian parameters
num_orbs = 24

# Retrieve Boston job data
print("Connecting to IBM Quantum...")
service = QiskitRuntimeService(name='qdc-2025')

job_id = 'd4b6vi7nmdfs73adjbs0'  # Boston
print(f"Retrieving job {job_id}...")
job = service.job(job_id)
job_result = job.result()

print(f"Job status: {job.status()}")
print(f"Number of circuit results: {len(job_result)}")

if len(job_result) > 0:
    # Get measurement data key
    meas_key = list(job_result[0].data.keys())[0]
    print(f"Measurement key: {meas_key}")
    
    # Combine the counts from individual Krylov circuits
    bit_array = BitArray.concatenate_shots(
        [j.data[meas_key] for j in job_result]
    )
    
    # Save to npz file
    output_file = 'bit_array_boston.npz'
    np.savez(output_file, num_bits=num_orbs * 2, samples=bit_array.array)
    print(f"\nSaved {output_file}")
    print(f"  Shape: {bit_array.array.shape}")
    print(f"  Num bits: {num_orbs * 2}")
else:
    print("ERROR: No results found in job!")
