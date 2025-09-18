#!/usr/bin/env python3
"""
SQD Test with Systematic Configuration Generation
"""

import argparse
import tomllib
from pathlib import Path
from typing import Optional
from itertools import combinations

# pyscf - python module for quantum chemistry - https://github.com/pyscf/pyscf
from pyscf import ao2mo, tools, scf, gto, fci

import matplotlib.pyplot as plt
import numpy as np

# IBM Qiskit addon library for SQD
from qiskit_addon_sqd.counts import generate_counts_uniform
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_fermion

from lwfm.base.Workflow import Workflow
from lwfm.midware.LwfManager import lwfManager, logger


def make_rhf(atom: str, basis: str, symmetry: bool, spin: int, charge: int,
    num_alpha: int, num_beta: int, dump_integrals: Optional[str] = None):
    """
    Take the description of an atom & basis set, run a restricted Hartree–Fock (RHF)
    calculation, returning the converged mean‑field object.
    """
    # Define the molecule
    print(f"{atom} basis {basis} spin {spin} charge {charge} symmetry {symmetry}")
    mol = gto.M(atom=atom, basis=basis, symmetry=symmetry, spin=spin, charge=charge)
    # Run a Restricted Hartree-Fock (RHF) calculation
    mf = scf.RHF(mol)
    mf.kernel()

    # Compute ground state energy using FCI
    total_electrons = num_alpha + num_beta
    
    print(f"Running FCI with {total_electrons} electrons (molecule has {mol.nelectron})")
    
    if total_electrons != mol.nelectron:
        print(f"WARNING: Electron count mismatch! SQD expects {total_electrons}, molecule has {mol.nelectron}")
        print("This will likely give poor results. Consider using a molecule with exactly {total_electrons} electrons.")
    
    # Always use full FCI for now to avoid active space complications
    cisolver = fci.FCI(mol, mf.mo_coeff)
    e_fci, _ = cisolver.kernel()

    print(f"SCF energy: {mf.e_tot:.10f} Ha")
    print(f"FCI ground state energy: {e_fci:.10f} Ha")
    print(f"Correlation energy: {e_fci - mf.e_tot:.10f} Ha")
    print(f"Active electrons: {total_electrons} / {mol.nelectron} total")

    if dump_integrals is not None:
        tools.fcidump.from_scf(mf, dump_integrals)
        print(f"FCIDUMP written to {dump_integrals}")

    # Store the ground state energy in the mf object for later use
    return  mf, e_fci


def generate_systematic_counts(num_orbitals: int, num_alpha: int, num_beta: int):
    """Generate all possible electron configurations systematically"""
    
    print("=== SYSTEMATIC CONFIGURATION GENERATION ===")
    print(f"Orbitals: {num_orbitals}, Alpha electrons: {num_alpha}, Beta electrons: {num_beta}")
    
    # Generate all possible alpha and beta configurations
    alpha_configs = list(combinations(range(num_orbitals), num_alpha))
    beta_configs = list(combinations(range(num_orbitals), num_beta))
    
    print(f"Alpha configurations: {len(alpha_configs)}")
    print(f"Beta configurations: {len(beta_configs)}")
    print(f"Total combinations: {len(alpha_configs) * len(beta_configs)}")
    
    # Create bitstrings for all combinations
    bitstrings = []
    for alpha_occ in alpha_configs:
        for beta_occ in beta_configs:
            # Create bitstring: first num_orbitals bits for alpha, next for beta
            bitstring = ['0'] * (num_orbitals * 2)
            for i in alpha_occ:
                bitstring[i] = '1'  # Alpha electrons
            for i in beta_occ:
                bitstring[i + num_orbitals] = '1'  # Beta electrons
            bitstrings.append(''.join(bitstring))
    
    # Convert to counts format (each configuration appears multiple times for better sampling)
    counts = {}
    for bs in bitstrings:
        counts[bs] = counts.get(bs, 0) + 1000  # Give each config weight of 1000 for better statistics
    
    print(f"Created systematic counts with {len(counts)} unique configurations")
    print("First few configurations:")
    for i, (bs, count) in enumerate(list(counts.items())[:3]):
        alpha_part = bs[:num_orbitals]
        beta_part = bs[num_orbitals:]
        print(f"  α:{alpha_part} β:{beta_part} (weight: {count})")
    
    return counts


def run_sqd_pipeline(
    molecule_scf: gto.M,
    g_energy: float,
    num_alpha: int,
    num_beta: int,
    open_shell: bool,
    spin_sq: int,
    iterations_count: int,
    num_batches: int,
    samples_per_batch: int,
    rng_seed: Optional[int] = 42,
    ):
    """
    Run the SQD pipeline and return data needed for plotting.
    """

    # Make in molecule and mine it for integrals
    num_orbitals = molecule_scf.mol.nao
    print(f"Number of orbitals: {num_orbitals}")
    # Core Hamiltonian representing the single-electron integrals
    core_hamiltonian = molecule_scf.get_hcore()
    print("Core Hamiltonian created...")
    # Electron repulsion integrals representing the two-electron integrals
    electron_repulsion_integrals = ao2mo.restore(1, molecule_scf._eri, num_orbitals)
    print("Electron Repulsion Integrals created...")

    # Nuclear repulsion energy of the molecule
    nuclear_repulsion_energy = molecule_scf.mol.energy_nuc()
    print(f"Nuclear Repulsion Energy: {nuclear_repulsion_energy}")

    # Generate systematic counts for small systems
    rng = np.random.default_rng(rng_seed)
    print(f"Generating counts for {num_orbitals * 2} qubits, targeting {num_alpha}α + {num_beta}β electrons")
    
    if num_orbitals <= 8:  # Use systematic for small systems
        counts = generate_systematic_counts(num_orbitals, num_alpha, num_beta)
    else:
        print("Large system - using uniform random sampling with Hamming weight bias")
        # For larger systems, generate many samples and rely on post-selection
        counts = generate_counts_uniform(500_000, num_orbitals * 2, rand_seed=rng)

    # Convert counts into bitstring and probability arrays
    bitstring_matrix_full, probs_array_full = counts_to_arrays(counts)

    # Initialize histories (use local names to avoid shadowing globals)
    energy_hist_local = np.zeros((iterations_count, num_batches))   # energy history
    spin_sq_hist_local = np.zeros((iterations_count, num_batches))  # spin history
    occupancy_hist_local: list = []
    avg_occupancy = None

    for i in range(iterations_count):
        print(f"\n=== Configuration recovery iteration {i+1}/{iterations_count} ===")
        print(f"Available configurations: {len(bitstring_matrix_full)}")
        
        # On the first iteration, we have no orbital occupancy information from the
        # solver, so we just post-select from the full bitstring set based on Hamming weight.
        if avg_occupancy is None:
            print("First iteration: using full bitstring set")
            bitstring_matrix_tmp = bitstring_matrix_full
            probs_array_tmp = probs_array_full
        else:
            print(f"Using recovered configurations based on occupancy")
            # If there is average orbital occupancy information, use it to refine the set
            bitstring_matrix_tmp, probs_array_tmp = recover_configurations(
                bitstring_matrix_full,
                probs_array_full,
                avg_occupancy,
                num_alpha,
                num_beta,
                rand_seed=rng,
            )
            print(f"Recovered configurations: {len(bitstring_matrix_tmp)}")

        # Post-select by desired particle numbers and then subsample
        bitstring_matrix_ps, probs_array_ps = postselect_by_hamming_right_and_left(
            bitstring_matrix_tmp,
            probs_array_tmp,
            hamming_right=num_alpha,
            hamming_left=num_beta,
        )
        print(f"  After post-selection: {len(bitstring_matrix_ps)} configurations")
        
        # Check if we have enough configurations
        if len(bitstring_matrix_ps) < 100:
            print(f"  WARNING: Only {len(bitstring_matrix_ps)} configurations available!")
        
        batches = subsample(
            bitstring_matrix_ps,
            probs_array_ps,
            samples_per_batch=min(samples_per_batch, len(bitstring_matrix_ps)),  # Don't exceed available
            num_batches=num_batches,
            rand_seed=rng,
        )

        # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
        e_tmp = np.zeros(num_batches)
        s_tmp = np.zeros(num_batches)
        occs_tmp = []
        coeffs = []
        for j in range(num_batches):
            strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
            print(f"Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")
            energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
                batches[j],
                core_hamiltonian,
                electron_repulsion_integrals,
                open_shell=open_shell,
                spin_sq=spin_sq,
            )
            energy_sci += nuclear_repulsion_energy
            e_tmp[j] = energy_sci
            s_tmp[j] = spin
            occs_tmp.append(avg_occs)
            coeffs.append(coeffs_sci)

        # Combine batch results
        avg_occupancy = tuple(np.mean(occs_tmp, axis=0))

        # Track optimization history
        energy_hist_local[i, :] = e_tmp
        spin_sq_hist_local[i, :] = s_tmp
        occupancy_hist_local.append(avg_occupancy)
        
        # Debug output
        current_energy = np.mean(e_tmp)
        energy_error = abs(current_energy - g_energy)
        print(f"Iteration {i+1} results:")
        print(f"  Average energy: {current_energy:.8f} Ha")
        print(f"  Reference (FCI): {g_energy:.8f} Ha") 
        print(f"  Energy error: {energy_error:.8f} Ha ({energy_error*27.2114:.4f} eV)")
        
        # Handle different occupancy array structures
        try:
            if hasattr(avg_occupancy[0], '__len__'):
                # If occupancy is nested (alpha, beta arrays)
                occ_str = f"α: {[f'{float(x):.3f}' for x in avg_occupancy[0]]}, β: {[f'{float(x):.3f}' for x in avg_occupancy[1]]}"
            else:
                # If occupancy is flat array
                occ_str = f"{[f'{float(x):.3f}' for x in avg_occupancy]}"
        except (TypeError, IndexError):
            # Fallback for any other structure
            occ_str = str(avg_occupancy)
        print(f"  Average occupancy: {occ_str}")
        
        if i > 0:
            prev_energy = np.mean(energy_hist_local[i-1, :])
            improvement = abs(prev_energy - g_energy) - energy_error
            print(f"  Improvement from last iteration: {improvement:.8f} Ha")
            if improvement < 1e-8:
                print("  WARNING: No significant improvement!")

    return (
        energy_hist_local,
        spin_sq_hist_local,
        occupancy_hist_local,
        iterations_count,
        g_energy,
    )


def plot_sqd_results(energy_hist_arr, occupancy_hist_list, iterations_for_plot, exact_energy_ref, output_path, show_plot=True):
    """Plot SQD results"""
    print(f"Plotting results to {output_path}")
    
    # Simple plot for now
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    iterations = range(1, iterations_for_plot + 1)
    energies = [np.mean(energy_hist_arr[i, :]) for i in range(iterations_for_plot)]
    errors = [abs(e - exact_energy_ref) for e in energies]
    
    ax.semilogy(iterations, errors, 'bo-', label='Energy Error')
    ax.axhline(y=1e-3, color='r', linestyle='--', label='1 mHa threshold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy Error (Ha)')
    ax.set_title('SQD Convergence')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return output_path


if __name__ == '__main__':
    # parse CLI args for workflow input file (required)
    parser = argparse.ArgumentParser(description="Run SQD cases from a TOML definition")
    parser.add_argument("workflow_toml", metavar="WORKFLOW_TOML",
                        help="Path to TOML file")
    args = parser.parse_args()

    wf_toml_path = Path(args.workflow_toml)
    if not wf_toml_path.exists():
        print(f"Error: Workflow TOML file not found: {wf_toml_path}")
        exit(1)

    # read the workflow TOML file
    with open(wf_toml_path, 'rb') as f:
        data = tomllib.load(f)

    globalArgs = data["global"]

    # make a workflow to bundle all these cases
    wf = Workflow("SQD variations", "Fun with SQD", {})
    wf = lwfManager.putWorkflow(wf)
    if wf is None:
        print("Error: Failed to register workflow.")
        exit(1)

    keepSaveDir = globalArgs["savedir"]

    # iterate over all the cases in the TOML file
    for caseId in data.keys():
        if caseId == "global":
            continue

        print(f"\n{'='*60}")
        print(f"Processing case: {caseId}")
        print(f"{'='*60}")

        caseArgs = data[caseId].copy()
        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId
        caseArgs.update(globalArgs)

        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)
        dump_fci_path = caseOutDir / "FCIDUMP"

        molecule_scf, g_energy = make_rhf(caseArgs['atom'], caseArgs['basis'],
            caseArgs['symmetry'], caseArgs['spin'], caseArgs['charge'],
            caseArgs['num_alpha'], caseArgs['num_beta'],
            dump_integrals=str(dump_fci_path) if caseArgs['dump_integrals'] else None)

        energy_hist, spin_sq_hist, occupancy_hist, iters_used, exact_energy = run_sqd_pipeline(
            molecule_scf=molecule_scf,
            g_energy=g_energy,
            num_alpha = caseArgs['num_alpha'],
            num_beta = caseArgs['num_beta'],
            open_shell = caseArgs['open_shell'],
            spin_sq = caseArgs['spin_sq'],
            iterations_count = caseArgs['iterations_count'],
            num_batches = caseArgs['num_batches'],
            samples_per_batch = caseArgs['samples_per_batch'],
            rng_seed = caseArgs['rng_seed']
        )

        output_path='/tmp/qt04_examples_sqd_test01.png'

        output_path = plot_sqd_results(
            energy_hist_arr=energy_hist,        # Energies per iteration and batch
            occupancy_hist_list=occupancy_hist, # Average (alpha, beta) occupancies
            iterations_for_plot=iters_used,     # Number of iterations to show
            exact_energy_ref=exact_energy,      # Reference exact energy
            output_path=output_path,            # Where to save the generated figure
            show_plot=True                      # If True, open image viewer after saving
        )

        print(f"\nFinal Results for {caseId}:")
        print(f"Exact energy: {exact_energy:.5f} Ha")
        print(f"SQD energy: {np.mean(energy_hist[-1, :]):.5f} Ha")
        print(f"Absolute error: {abs(np.mean(energy_hist[-1, :]) - exact_energy):.5f} Ha")
