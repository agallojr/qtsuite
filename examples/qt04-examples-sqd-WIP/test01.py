"""
test01 - Sample-based Quantum Diagonalization (SQD) - toy example
"""

from pyscf import ao2mo, tools
import matplotlib.pyplot as plt
import subprocess

from qiskit_addon_sqd.counts import generate_counts_uniform
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_and_subsample
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    solve_fermion,
)

import numpy as np


# Specify molecule properties
num_orbitals = 16
num_alpha = num_beta = 5
open_shell = False
spin_sq = 0
 
# Read in molecule from disk
molecule_scf = tools.fcidump.to_scf("n2_fci.txt")
 
# Core Hamiltonian representing the single-electron integrals
core_hamiltonian = molecule_scf.get_hcore()
 
# Electron repulsion integrals representing the two-electron integrals
electron_repulsion_integrals = ao2mo.restore(
    1, molecule_scf._eri, num_orbitals
)
 
# Nuclear repulsion energy of the molecule
nuclear_repulsion_energy = molecule_scf.mol.energy_nuc()

print("Core Hamiltonian:")
print(core_hamiltonian)
print("\nElectron Repulsion Integrals:")
print(electron_repulsion_integrals)
print("\nNuclear Repulsion Energy:")
print(nuclear_repulsion_energy)

# The artificial generation of samples is only used as a means to demonstrate the tooling of 
# qiskit-addon-sqd. The package is meant to be used as part of a larger workflow wherein an ansatz # or other circuit is defined, optimized, and executed using the Sampler primitive. The code cell 
# below contains commented-out code demonstrating what a typical workflow might look like given a 
# circuit ansatz transpiled to an ISA circuit.

# from qiskit_ibm_runtime import SamplerV2 as Sampler
 
# sampler = Sampler(mode=backend)
# job = sampler.run([isa_circuit], shots=10_000)
# primitive_result = job.result()
# pub_result = primitive_result[0]
# counts = pub_result.data.meas.get_counts()
 

 
rng = np.random.default_rng(24)
counts = generate_counts_uniform(10_000, num_orbitals * 2, rand_seed=rng)
 
# Convert counts into bitstring and probability arrays
bitstring_matrix_full, probs_array_full = counts_to_arrays(counts)

# SQD options
ITERATIONS = 5
 
# Eigenstate solver options
NUM_BATCHES = 1
SAMPLES_PER_BATCH = 500
MAX_DAVIDSON_CYCLES = 200

# Self-consistent configuration recovery loop
energy_hist = np.zeros((ITERATIONS, NUM_BATCHES))  # energy history
spin_sq_hist = np.zeros((ITERATIONS, NUM_BATCHES))  # spin history
occupancy_hist = []
avg_occupancy = None


 
for i in range(ITERATIONS):
    print(f"\nStarting configuration recovery iteration {i}")
    # On the first iteration, we have no orbital occupancy information from the
    # solver, so we just post-select from the full bitstring set based on Hamming weight.
    if avg_occupancy is None:
        bitstring_matrix_tmp = bitstring_matrix_full
        probs_array_tmp = probs_array_full
 
    # If there is average orbital occupancy information, use it to refine the full set of noisy configurations
    else:
        bitstring_matrix_tmp, probs_array_tmp = recover_configurations(
            bitstring_matrix_full,
            probs_array_full,
            avg_occupancy,
            num_alpha,
            num_beta,
            rand_seed=rng,
        )
 
    # Throw out configurations with an incorrect particle number in either the spin-up or spin-down systems
    batches = postselect_and_subsample(
        bitstring_matrix_tmp,
        probs_array_tmp,
        hamming_right=num_alpha,
        hamming_left=num_beta,
        samples_per_batch=SAMPLES_PER_BATCH,
        num_batches=NUM_BATCHES,
        rand_seed=rng,
    )
 
    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    e_tmp = np.zeros(NUM_BATCHES)
    s_tmp = np.zeros(NUM_BATCHES)
    occs_tmp = []
    coeffs = []
    for j in range(NUM_BATCHES):
        strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
        print(f"Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")
        energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
            batches[j],
            core_hamiltonian,
            electron_repulsion_integrals,
            open_shell=open_shell,
            spin_sq=spin_sq,
            max_davidson=MAX_DAVIDSON_CYCLES,
        )
        energy_sci += nuclear_repulsion_energy
        e_tmp[j] = energy_sci
        s_tmp[j] = spin
        occs_tmp.append(avg_occs)
        coeffs.append(coeffs_sci)
 
    # Combine batch results
    avg_occupancy = tuple(np.mean(occs_tmp, axis=0))
 
    # Track optimization history
    energy_hist[i, :] = e_tmp
    spin_sq_hist[i, :] = s_tmp
    occupancy_hist.append(avg_occupancy)
    print()

    # Chemical accuracy (+/- 1 milli-Hartree)
chem_accuracy = 0.001
exact_energy = -109.04667
 
# Data for energies plot
x1 = range(ITERATIONS)
min_e = [np.min(e) for e in energy_hist]
e_diff = [abs(e - exact_energy) for e in min_e]
yt1 = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
 
# Data for avg spatial orbital occupancy
y2 = occupancy_hist[-1][0] + occupancy_hist[-1][1]
x2 = range(len(y2))
 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
 
# Plot energies
axs[0].plot(x1, e_diff, label="energy error", marker="o")
axs[0].set_xticks(x1)
axs[0].set_xticklabels(x1)
axs[0].set_yticks(yt1)
axs[0].set_yticklabels(yt1)
axs[0].set_yscale("log")
axs[0].set_ylim(1e-4)
axs[0].axhline(
    y=chem_accuracy,
    color="#BF5700",
    linestyle="--",
    label="chemical accuracy",
)
axs[0].set_title("Approximated Ground State Energy Error vs SQD Iterations")
axs[0].set_xlabel("Iteration Index", fontdict={"fontsize": 12})
axs[0].set_ylabel("Energy Error (Ha)", fontdict={"fontsize": 12})
axs[0].legend()
 
# Plot orbital occupancy
axs[1].bar(x2, y2, width=0.8)
axs[1].set_xticks(x2)
axs[1].set_xticklabels(x2)
axs[1].set_title("Avg Occupancy per Spatial Orbital")
axs[1].set_xlabel("Orbital Index", fontdict={"fontsize": 12})
axs[1].set_ylabel("Avg Occupancy", fontdict={"fontsize": 12})
 
print(f"Exact energy: {exact_energy:.5f} Ha")
print(f"SQD energy: {min_e[-1]:.5f} Ha")
print(f"Absolute error: {e_diff[-1]:.5f} Ha")
plt.tight_layout()

plt.savefig('/tmp/qt04_examples_sqd.png', bbox_inches='tight')
subprocess.Popen(['eog', '/tmp/qt04_examples_sqd.png'])
