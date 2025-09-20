"""
test01 - Sample-based Quantum Diagonalization (SQD) 
"Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer"
arXiv:2405.05068
"""

#pylint: disable=invalid-name, protected-access, consider-using-with, redefined-outer-name
#pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals

import subprocess
import tomllib
import argparse
from pathlib import Path
import sys

# pyscf - python module for quantum chemistry - https://github.com/pyscf/pyscf
from pyscf import ao2mo, tools

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

# IBM Qiskit addon library for SQD
from qiskit_addon_sqd.counts import generate_counts_uniform
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_fermion

from lwfm.base.Workflow import Workflow
from lwfm.midware.LwfManager import lwfManager, logger


def run_sqd_pipeline(
    num_orbitals: int,
    core_hamiltonian: np.ndarray,
    electron_repulsion_integrals: np.ndarray,
    nuclear_repulsion_energy: float,
    num_alpha: int,
    num_beta: int,
    open_shell: bool,
    spin_sq: int,
    iterations_count: int,
    num_batches: int,
    samples_per_batch: int,
    rng_seed: int | None = 42,
    ):
    """
    Run the SQD pipeline and return data needed for plotting.

    Returns
    -------
    energy_hist: np.ndarray
        Energies per iteration and batch (Hartree), including nuclear repulsion.
    spin_sq_hist: np.ndarray
        Spin expectation per iteration and batch.
    occupancy_hist: list[tuple[np.ndarray, np.ndarray]]
        Average occupancies for alpha/beta per iteration.
    iterations: int
        Number of iterations used (for plotting convenience).
    exact_energy: float
        Reference energy used to compute errors.
    """

    # Generate synthetic counts
    # TODO: replace with real counts from hardware
    # from qiskit_ibm_runtime import SamplerV2 as Sampler
    
    # sampler = Sampler(mode=backend)
    # job = sampler.run([isa_circuit], shots=10_000)
    # primitive_result = job.result()
    # pub_result = primitive_result[0]
    # counts = pub_result.data.meas.get_counts()
    rng = np.random.default_rng(rng_seed)
    counts = generate_counts_uniform(10_000, num_orbitals * 2, rand_seed=rng)

    # Convert counts into bitstring and probability arrays
    bitstring_matrix_full, probs_array_full = counts_to_arrays(counts)

    # Initialize histories (use local names to avoid shadowing globals)
    energy_hist_local = np.zeros((iterations_count, num_batches))   # energy history
    spin_sq_hist_local = np.zeros((iterations_count, num_batches))  # spin history
    occupancy_hist_local: list = []
    avg_occupancy = None

    for i in range(iterations_count):
        print(f"Starting configuration recovery iteration {i}")
        # On the first iteration, we have no orbital occupancy information from the
        # solver, so we just post-select from the full bitstring set based on Hamming weight.
        if avg_occupancy is None:
            bitstring_matrix_tmp = bitstring_matrix_full
            probs_array_tmp = probs_array_full
        else:
            # If there is average orbital occupancy information, use it to refine the set
            bitstring_matrix_tmp, probs_array_tmp = recover_configurations(
                bitstring_matrix_full,
                probs_array_full,
                avg_occupancy,
                num_alpha,
                num_beta,
                rand_seed=rng,
            )

        # Post-select by desired particle numbers and then subsample
        bitstring_matrix_ps, probs_array_ps = postselect_by_hamming_right_and_left(
            bitstring_matrix_tmp,
            probs_array_tmp,
            hamming_right=num_alpha,
            hamming_left=num_beta,
        )
        batches = subsample(
            bitstring_matrix_ps,
            probs_array_ps,
            samples_per_batch=samples_per_batch,
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

    return (
        energy_hist_local,
        spin_sq_hist_local,
        occupancy_hist_local
    )


def plot_sqd_results(
    energy_hists: dict[str, np.ndarray],
    iterations: dict[str, int],
    exact_energies: dict[str, float],
    output_path: str,
    show_plot: bool
    ) -> str:
    """
    Plot SQD energy error convergence for multiple cases, save to file, [and open viewer]

    Parameters
    ----------
    energy_hists : dict[str, np.ndarray]
        Dictionary mapping caseId to energy history arrays
    iterations : dict[str, int]
        Dictionary mapping caseId to number of iterations
    exact_energies : dict[str, float]
        Dictionary mapping caseId to exact reference energy
    output_path : str
        Path where to save the generated figure
    show_plot : bool
        Whether to display the plot after saving

    Returns
    -------
    str
        The file path of the saved figure.
    """

    # Generate unique colors for each case
    case_ids = list(energy_hists.keys())
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan']
    colors = [color_list[i % len(color_list)] for i in range(len(case_ids))]

    yt1 = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
    chem_accuracy = 0.001  # Chemical accuracy (+/- 1 milli-Hartree)

    _fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    # Plot energies for each case
    for i, case_id in enumerate(case_ids):
        energy_hist = energy_hists[case_id]
        exact_energy = exact_energies[case_id]
        num_iterations = iterations[case_id]

        x = range(num_iterations)
        min_e = [np.min(e) for e in energy_hist[:num_iterations]]
        e_diff = [abs(e - exact_energy) for e in min_e]

        axs.plot(x, e_diff, label=f"{case_id}", marker="o", color=colors[i])

        # Print results for this case
        print(f"Case {case_id}:")
        print(f"  Exact energy: {exact_energy:.5f} Ha")
        print(f"  SQD energy: {min_e[-1]:.5f} Ha")
        print(f"  Absolute error: {e_diff[-1]:.5f} Ha")

    axs.set_yticks(yt1)
    axs.set_yticklabels(yt1)
    axs.set_yscale("log")
    axs.set_ylim(1e-4)
    axs.axhline(
        y=chem_accuracy,
        color="#BF5700",
        linestyle="--",
        label="chemical accuracy",
    )
    axs.set_title("Approximated Ground State Energy Error vs SQD Iterations")
    axs.set_xlabel("Iteration Index", fontdict={"fontsize": 12})
    axs.set_ylabel("Energy Error (Ha)", fontdict={"fontsize": 12})
    axs.legend()
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    if show_plot:
        subprocess.Popen(['eog', output_path])
    return output_path


if __name__ == '__main__':

    # *******************************************
    # parse CLI args for workflow input file (required)

    parser = argparse.ArgumentParser(description="Run SQD cases from a TOML definition")
    parser.add_argument("workflow_toml", metavar="WORKFLOW_TOML",
                        help="Path to TOML file")
    parser.add_argument("--case", metavar="CASE_NAME",
                        help="Specific case to run (default: run all cases)")
    args = parser.parse_args()

    wf_toml_path = Path(args.workflow_toml)
    if not wf_toml_path.is_file():
        print(f"Error: {wf_toml_path} not found.")
        sys.exit(1)

    # load the TOML file of test case inputs
    try:
        with open(wf_toml_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error decoding TOML: {e}")
        sys.exit(1)
    globalArgs = data["global"]

    # *******************************************
    # make a workflow to bundle the cases

    wf = Workflow("SQD variations", "Fun with SQD", {})
    wf = lwfManager.putWorkflow(wf)
    if wf is None:
        print("Error: Failed to register workflow.")
        sys.exit(1)
    logger.setContext(wf)
    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())
    # we're going to modify the savedir for each case, so keep a copy of the original
    keepSaveDir = globalArgs["savedir"]

    # warm up sandboxes we use - here the latest Qiskit libs
    # lwfManager.updateSite("ibm-quantum-venv")

    # *******************************************
    # for each case in the workflow toml

    cases_to_run = ((k, v) for k, v in data.items() if k != "global")
    if args.case:
        cases_to_run = ((k, v) for k, v in data.items() if k == args.case)

    energy_hists = {}
    iterations = {}
    exact_energies = {}

    for caseId, caseArgs in cases_to_run:
        print(f"Running case: {caseId}")
        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId + "/"
        caseArgs.update(globalArgs)

        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)

        # Read in molecule from disk and mine it for integrals
        molecule_scf = tools.fcidump.to_scf(caseArgs['input_fcidump'])
        print("Molecule read from file.")
        # Core Hamiltonian representing the single-electron integrals
        core_hamiltonian = molecule_scf.get_hcore()
        print("Core Hamiltonian created.")
        # Electron repulsion integrals representing the two-electron integrals
        electron_repulsion_integrals = ao2mo.restore(1, molecule_scf._eri,
            caseArgs['num_orbitals'])
        print("Electron Repulsion Integrals created.")

        # Nuclear repulsion energy of the molecule
        nuclear_repulsion_energy = molecule_scf.mol.energy_nuc()
        print(f"\nNuclear Repulsion Energy: {nuclear_repulsion_energy}\n")

        energy_hist, spin_sq_hist, occupancy_hist = run_sqd_pipeline(
            core_hamiltonian = core_hamiltonian,
            electron_repulsion_integrals = electron_repulsion_integrals,
            nuclear_repulsion_energy = nuclear_repulsion_energy,
            num_orbitals = caseArgs['num_orbitals'],
            num_alpha = caseArgs['num_alpha'],
            num_beta = caseArgs['num_beta'],
            open_shell = caseArgs['open_shell'],
            spin_sq = caseArgs['spin_sq'],
            iterations_count = caseArgs['iterations_count'],
            num_batches = caseArgs['num_batches'],
            samples_per_batch = caseArgs['samples_per_batch'],
            rng_seed = caseArgs['rng_seed']
        )
        energy_hists[caseId] = energy_hist
        iterations[caseId] = caseArgs['iterations_count']
        exact_energies[caseId] = caseArgs['exact_energy_ref']


    output_path=globalArgs["savedir"] + "result.png"
    output_path = plot_sqd_results(
        energy_hists=energy_hists,        # Energies per iteration and batch (Hartree)
        iterations=iterations, # for x-axis
        exact_energies=exact_energies,    # reference exact energy
        output_path=output_path,            # where to save the generated figure
        show_plot=True
    )
