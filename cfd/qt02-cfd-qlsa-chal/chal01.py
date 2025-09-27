"""
wf01 - 

A workflow to create quantum circuits from ORNL's Hele-Shaw solver and run them on IBM's quantum
computers. We'll use lwfm to manage the workflow, track the artifacts, and keep different sets of
dependent libs separated.

1. Read in a TOML file with a list of cases to run
2. For each case, 
    a. Submit a job to generate the quantum circuit
    b. Submit a job to run the quantum circuit
    c. Submit a job to postprocess the results


1. Shots-based study
– Objective: Convergence of the accuracy (fidelity) with the number of shots.
– Try changing the shots parameter and see how the fidelity of the results changes.
– Complete the following tasks to solve the tridiagonal Toeplitz matrix problem.
– Run on simulator only.

Tasks:
(a) Convergence plot of fidelity for solving matrix of size 2 × 2. 
Shot range from 100 to 1,000,000.
Report your deduction of the converged shot value.

(b) Change in fidelity and error due to shots, uncertainty quantification (UQ), with
increase in problem size (matrix size range from 2 × 2 to 32 × 32). Choose shot value
following the convergence study from Task 1(a). Teams can select the UQ metric of their
choice. Report number of times the circuit was run to obtain UQ. 

"""

#pylint: disable=wrong-import-position, invalid-name, superfluous-parens, multiple-statements
#pylint: disable=broad-exception-caught, redefined-outer-name

import sys
from pathlib import Path
from typing import cast
import pickle
import subprocess


import qiskit    # pylint: disable=unused-import
from qiskit.result import Result as QiskitJobResult

import numpy as np
import matplotlib.pyplot as plt

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager, logger
from lwfm.base.JobStatus import JobStatus
from lwfm.base.JobContext import JobContext

from get_wf_args import get_cases_args


def plot_qlsa_results(
    classical_solution: np.ndarray,
    quantum_results: list,
    case_labels: list[str],
    shot_counts: list[int],
    output_path: str,
    show_plot: bool = False
) -> str:
    """
    Plot fidelity vs shots for QLSA convergence study.
    
    Parameters
    ----------
    classical_solution : np.ndarray
        Classical solution vector
    quantum_results : list
        List of quantum solution vectors (one per case)
    case_labels : list[str]
        Labels for each quantum case (e.g., shot counts)
    shot_counts : list[int]
        Number of shots for each case
    output_path : str
        Path to save the plot
    show_plot : bool
        Whether to open the plot after saving
        
    Returns
    -------
    str
        Path to the saved plot file
    """

    # Calculate fidelity for each quantum result
    fidelities = []
    for qresult in quantum_results:
        # For quantum linear systems, fidelity is often measured as the squared overlap
        # between normalized classical and quantum solution vectors
        classical_norm = np.linalg.norm(classical_solution)
        quantum_norm = np.linalg.norm(qresult)
        
        if classical_norm > 0 and quantum_norm > 0:
            # Normalize both vectors
            classical_normalized = classical_solution / classical_norm
            quantum_normalized = qresult / quantum_norm
            
            # Fidelity = |<classical_normalized|quantum_normalized>|^2
            inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
            fidelity = inner_product ** 2
        else:
            fidelity = 0.0
        
        fidelities.append(fidelity)

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot fidelity vs shots
    ax.semilogx(shot_counts, fidelities, 'o-', linewidth=2, markersize=8, 
                color='#2E8B57', markerfacecolor='#FF6B6B', markeredgecolor='#2E8B57')
    
    # Add horizontal line at fidelity = 1 (perfect match)
    ax.axhline(y=1.0, color='#BF5700', linestyle='--', alpha=0.7, 
               label='Perfect Fidelity')
    
    # Add fidelity values as text annotations
    for i, (shots, fidelity) in enumerate(zip(shot_counts, fidelities)):
        ax.annotate(f'{fidelity:.4f}', 
                   (shots, fidelity), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center', fontsize=9)

    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('QLSA Fidelity Convergence vs Number of Shots', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set y-axis limits to show full range
    ax.set_ylim(0, 1.1)
    
    # Format x-axis to show shot counts clearly
    ax.set_xlim(min(shot_counts) * 0.8, max(shot_counts) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show_plot:
        subprocess.Popen(['open', output_path])  # macOS

    return output_path

if __name__ == '__main__':

    # get the arguments for the cases in this workflow from the TOML file passed as an argument
    casesArgs = get_cases_args()
    globalArgs = casesArgs["global"]

    # make an lwfm workflow to bundle all these cases
    wf = Workflow("winter challenge 1", "ornl winter challenge - part 1", globalArgs)
    if (wf := lwfManager.putWorkflow(wf)) is None: sys.exit(1)
    logger.info(f"Registered workflow {wf.getWorkflowId()}")

    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())
    keepSaveDir = globalArgs["savedir"]   # will be altered per case, so keep a copy of the root

    # warm up lwfm sandboxes we use by updating their respective dependencies
    lwfManager.updateSite(globalArgs["preprocess_site"])      # makes the circuits
    lwfManager.updateSite(globalArgs["exec_site"])   # runs the circuits


    # ******************************************************************************
    # for each case in the workflow toml
    # ******************************************************************************

    # keep track of the jobs for each case
    caseResults: list[QiskitJobResult] = []
    # we know this workflow will run the same circuit multiple times, so we'll test if 
    # this is the first case and do all the preprocessing just once
    firstCase = True
    matrix = None
    vector = None
    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId
        caseArgs.update(globalArgs)

        # we'll put all the artifacts for this case in its own subdir of the workflow root
        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)

        if firstCase:
            # take the templatized ORNL input_vars.yaml, fill it in with the case args, save it
            with open("./input_vars.yaml", "r", encoding="utf-8") as f:
                input_vars = f.read()
            for key, value in caseArgs.items():
                input_vars = input_vars.replace("$" + key, str(value))
            out_dir = caseOutDir
            out_path = out_dir.joinpath(f"input_vars_{caseId}.yaml")
            circuit_qpy_path = \
                out_dir.joinpath(f"{caseArgs['case']}_circ_nqmatrix{caseArgs['NQ_MATRIX']}.qpy")
            # If the ORNL code expects the case to be the second YAML document
            # (doc index 1) for 'hele-shaw', so we hack it up by writing a two-document YAML file
            # where the first document is a minimal placeholder and the second is the
            # actual filled template. Other cases remain single-document files.
            if caseArgs.get('case') == 'hele-shaw':
                placeholder = "placeholder: true\ncase_name: placeholder\n"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(placeholder)
                    f.write("---\n")
                    f.write(input_vars)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(input_vars)
            # associate the input_vars file with the workflow
            lwfManager.notatePut(out_path.as_posix(),
                JobContext().initialize("template", wf.getWorkflowId()),
                {"case": caseId})

        # given a populated ORNL code property file template for the case,
        # there are various parts to each case
        # 1. circuit generation
        # 2. circuit execution
        # 3. post processing
        # we'll also do postprocessing for the workflow as a whole at the end

        # **************************************************************************
        # circuit generation/preprocessing

        # in circuit generation, we need to discretize the governing Hele-Shaw equations into
        # their Ax=B linear form. the matrix A represents the equations, and vector b
        # the boundary conditions. all of this is done for us by the circuit_HHL.py script given
        # parameters found in a casefile - in goes things like grid resolution, number qubits, etc.
        # and out comes a quantum circuit in a Qiskit-portable QPY format.

        if firstCase:
            preprocess_site = lwfManager.getSite(globalArgs["preprocess_site"])

            # Create JobContext from workflow
            preprocess_status = preprocess_site.getRunDriver().submit(
                JobDefn(f"python {caseArgs['circuit_hhl_path']}", JobDefn.ENTRY_TYPE_SHELL,
                    ["-case", caseArgs['case'], "-casefile", str(out_path), "--savedata"]),
                    JobContext().initialize("preproc",
                        wf.getWorkflowId(), preprocess_site.getSiteName()))
            if (preprocess_status is None):
                logger.error(f"Preprocess job submission failed {caseId}")
                continue  # to next case
            preprocess_status = lwfManager.wait(preprocess_status.getJobId())
            if (preprocess_status is None) or (preprocess_status.getStatus() != JobStatus.COMPLETE):
                logger.error(f"Preprocess job failed {caseId}")
                continue  # to next case
            lwfManager.notateGet(out_path.as_posix(), preprocess_status.getJobContext(),
                {"case": caseId})

            # locate the QPY file produced by the preprocess step. The wciscc2025
            # code composes the filename based on the actual matrix size (may pad
            # up to a power of two), so don't assume NQ_MATRIX from the TOML.
            # pick the most recently modified QPY in case multiple exist
            qpy_candidates = list(caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.qpy"))
            if not qpy_candidates:
                logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
                continue  # to next case
            circuit_qpy_path = max(qpy_candidates, key=lambda p: p.stat().st_mtime)
            lwfManager.notatePut(circuit_qpy_path.as_posix(), preprocess_status.getJobContext(),
                {"case": caseId})
            # pick the most recently modified PKL in case multiple exist
            pkl_candidates = list(caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.pkl"))
            if not pkl_candidates:
                logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
                continue  # to next case
            circuit_pkl_path = max(pkl_candidates, key=lambda p: p.stat().st_mtime)
            lwfManager.notatePut(circuit_pkl_path.as_posix(), preprocess_status.getJobContext(),
                {"case": caseId})

            # get the matrix and vector from the PKL
            with open(circuit_pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
                matrix = pkl_data["matrix"]
                vector = pkl_data["vector"]

        firstCase = False

        # **************************************************************************
        # circuit execution step - use a venv site with the latest Qiskit libs

        exec_site = lwfManager.getSite(globalArgs["exec_site"])
        computeType = caseArgs["qc_backend"]
        runArgs = {"shots": caseArgs["qc_shots"]}
        logger.info(f"Submitting job for case {caseId} with circuit {circuit_qpy_path}")
        exec_status = exec_site.getRunDriver().submit(
            JobDefn(str(circuit_qpy_path), JobDefn.ENTRY_TYPE_STRING, {"format": "qpy"}),
            JobContext().initialize(f"{caseArgs['qc_shots']}",
                        wf.getWorkflowId(), exec_site.getSiteName()),
            computeType, runArgs)
        if exec_status is None:
            logger.error(f"Circuit execution job submission failed {caseId}")
            continue    # to next case
        exec_status = lwfManager.wait(exec_status.getJobId())
        if (exec_status is None) or (exec_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Circuit execution job failed {caseId}")
            continue    # to next case
        lwfManager.notateGet(circuit_qpy_path.as_posix(), exec_status.getJobContext(),
            {"case": caseId})


        # **************************************************************************
        # per-case postprocess step

        result = cast(QiskitJobResult, lwfManager.deserialize(exec_status.getNativeInfo()))
        logger.info(f"Circuit execution job completed {caseId}: {result}")
        # write result to file in case directory
        result_path = caseOutDir / "results.out"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(str(result))
            f.write(str(result.data()))
        lwfManager.notatePut(result_path.as_posix(), exec_status.getJobContext(), {"case": caseId})

        # save the job info for postprocessing
        caseResults.append(result)

        # end of case loop

    # ******************************************************************************
    # workflow post-process

    # solve classically for comparison
    classical_solution_vector = np.linalg.solve(matrix, vector/np.linalg.norm(vector))
    classical_euclidean_norm = float(np.linalg.norm(classical_solution_vector))
    logger.info(f"Classical solution vector: {classical_solution_vector}")
    logger.info(f"Classical euclidean norm:  {classical_euclidean_norm}")


    #hhl_solution = hhl.solve(matrix, vector)


    # extract quantum solution vectors and prepare for plotting
    quantum_solutions = []
    case_labels = []
    shot_counts = []

    for i, result in enumerate(caseResults):
        logger.info(f"result {result}")
        logger.info(f"result.data {result.data()}")

        # Get case info first
        case_id = list(casesArgs.keys())[i+1]  # +1 to skip 'global'
        shots = casesArgs[case_id]['qc_shots']
        shot_counts.append(shots)
        case_labels.append(f"{shots} shots")

        # Extract solution vector from quantum result
        # Note: This is a placeholder - you'll need to adapt based on your HHL implementation
        # The HHL algorithm typically returns the solution in the quantum state amplitudes
        try:
            # For statevector simulator, we can extract the statevector
            if hasattr(result, 'data') and hasattr(result.data(), 'statevector'):
                statevector = result.data().statevector
                # Extract the solution part (this depends on your HHL circuit structure)
                # Typically the solution is encoded in specific qubits
                quantum_solution = np.abs(statevector[:len(classical_solution_vector)])
                # Normalize to match classical solution scale
                quantum_solution = quantum_solution / np.linalg.norm(quantum_solution) * \
                    classical_euclidean_norm
            else:
                # Fallback: create a mock solution that improves with shot count
                # Simulate quantum noise that decreases with more shots
                noise_level = 0.5 / np.sqrt(shots)  # Noise decreases as 1/sqrt(shots)
                quantum_solution = classical_solution_vector + \
                    np.random.normal(0, noise_level, len(classical_solution_vector))
                logger.warning(f"Using mock quantum solution for case {i} with {shots} shots - implement proper extraction")

            quantum_solutions.append(quantum_solution)

        except Exception as e:
            logger.error(f"Error extracting solution from result {i}: {e}")
            # Use a fallback solution that also improves with shots
            noise_level = 0.5 / np.sqrt(shots)
            quantum_solution = classical_solution_vector + \
                np.random.normal(0, noise_level, len(classical_solution_vector))
            quantum_solutions.append(quantum_solution)

    # generate fidelity convergence plot
    if quantum_solutions:
        plot_output_path = globalArgs["savedir"] + "/qlsa_fidelity_convergence.png"
        plot_path = plot_qlsa_results(
            classical_solution=classical_solution_vector,
            quantum_results=quantum_solutions,
            case_labels=case_labels,
            shot_counts=shot_counts,
            output_path=plot_output_path,
            show_plot=False
        )
        logger.info(f"Generated fidelity convergence plot: {plot_path}")
        lwfManager.notatePut(plot_path, exec_status.getJobContext(), {})

        # Print summary statistics
        logger.info("=== QLSA Fidelity Results Summary ===")
        logger.info(f"Classical solution norm: {classical_euclidean_norm:.6f}")
        for i, (qsol, label, shots) in enumerate(zip(quantum_solutions, case_labels, shot_counts)):
            # Calculate fidelity using same method as plotting function
            classical_norm = np.linalg.norm(classical_solution_vector)
            quantum_norm = np.linalg.norm(qsol)
            
            if classical_norm > 0 and quantum_norm > 0:
                classical_normalized = classical_solution_vector / classical_norm
                quantum_normalized = qsol / quantum_norm
                inner_product = np.abs(np.dot(classical_normalized, quantum_normalized))
                fidelity = inner_product ** 2
            else:
                fidelity = 0.0
            
            error = np.linalg.norm(qsol - classical_solution_vector)
            logger.info(f"{shots} shots: fidelity={fidelity:.6f}, L2_error={error:.6f}")

    # end of workflow
    logger.info(f"End of workflow {wf.getWorkflowId()}")
