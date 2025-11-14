"""
Serverless diagonalization engine for SQD computation.
This function is uploaded to Qiskit Serverless to run diagonalization remotely.
"""

import json
import numpy as np
from functools import partial
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)


def main(
    data: str,
    energy_tol: float = 1e-4,
    occupancies_tol: float = 1e-3,
    max_iterations: int = 12,
    num_batches: int = 8,
    samples_per_batch: int = 300,
    symmetrize_spin: bool = False,
    carryover_threshold: float = 1e-5,
    max_cycle: int = 200,
):
    """Main serverless function for diagonalization.
    
    Args:
        data: JSON-encoded list containing [job_id, hcore, eri, num_orbitals, num_elec_a, num_elec_b]
        energy_tol: SQD energy convergence tolerance
        occupancies_tol: SQD occupancies convergence tolerance
        max_iterations: Maximum SQD iterations
        num_batches: Number of batches for eigenstate solver
        samples_per_batch: Samples per batch
        symmetrize_spin: Whether to enforce spin symmetry
        carryover_threshold: Threshold for state carryover
        max_cycle: Maximum SCI solver cycles
        
    Returns:
        List of minimum energies for each iteration
    """
    # Decode input data
    data_list = json.loads(data)
    job_id, hcore_list, eri_list, num_orbitals, num_elec_a, num_elec_b = data_list
    
    # Convert lists back to numpy arrays
    hcore = np.array(hcore_list)
    eri = np.array(eri_list)
    
    print(f">>>>> Processing job {job_id}")
    print(f">>>>> Parameters: norb={num_orbitals}, nelec=({num_elec_a},{num_elec_b})")
    print(f">>>>> SQD: max_iter={max_iterations}, energy_tol={energy_tol}, occ_tol={occupancies_tol}")
    print(f">>>>> Solver: batches={num_batches}, samples={samples_per_batch}, max_cycle={max_cycle}")
    
    # Retrieve the quantum job results
    service = QiskitRuntimeService()
    quantum_job = service.job(job_id)
    print(f">>>>> Retrieved quantum job: {quantum_job.job_id()}")
    
    result = quantum_job.result()
    print(f">>>>> Got {len(result)} circuit results")
    
    # Get the measurement key (usually 'meas')
    meas_key = list(result[0].data.keys())[0]
    
    # Collect bit arrays from all circuits
    bit_arrays = [res.data[meas_key] for res in result]
    
    # Combine into single bit array
    from qiskit.primitives import BitArray
    bit_array = BitArray.concatenate_shots(bit_arrays)
    
    print(f">>>>> Combined bit array shape: {bit_array.array.shape}")
    
    # Configure SCI solver
    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)
    
    # List to capture intermediate results
    result_history = []
    
    def callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy}")
            print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")
    
    # Run diagonalization
    _ = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=num_orbitals,
        nelec=(num_elec_a, num_elec_b),
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        callback=callback,
        seed=12345,
    )
    
    # Extract minimum energies from each iteration
    min_es = [
        min(result, key=lambda res: res.energy).energy
        for result in result_history
    ]
    energy_history = [float(e) for e in min_es]
    
    print(f">>>>> Final energy history: {energy_history}")
    
    return energy_history


if __name__ == "__main__":
    # This allows the function to be called by Qiskit Serverless
    import sys
    if len(sys.argv) > 1:
        # Parse command line arguments passed by serverless
        result = main(*sys.argv[1:])
        print(json.dumps(result))
