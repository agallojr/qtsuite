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
#pylint: disable=broad-exception-caught, redefined-outer-name, consider-using-enumerate

import sys

from lwfm.midware.LwfManager import lwfManager, logger

from main_workflow import run_workflow
from plotting import plot_qlsa_generic
from plotting_scaling import plot_scaling_analysis, plot_scaling_table

if __name__ == '__main__':

    # Run the main workflow and get results
    (wf, caseResults, quantum_solutions, classical_solutions,
     casesArgs, globalArgs, exec_status) = run_workflow()

    # ******************************************************************************
    # workflow post-process

    logger.info("=" * 80)
    logger.info("Post-processing workflow results")
    logger.info(f"Number of cases with quantum solutions: {len(quantum_solutions)}")

    # Prepare data for plotting using already-extracted solutions
    case_labels = []
    case_data = []

    # Get case info for each result
    case_ids = [k for k in casesArgs.keys() if k != 'global']
    for i, case_id in enumerate(case_ids):
        case_params = casesArgs[case_id]

        # Extract metadata if available
        metadata = case_params.get('_metadata', {})

        # Build case data dict
        case_info = {
            'case_id': case_id,
            'params': case_params,
            'metadata': metadata,
            'classical_solution': classical_solutions[i] if i < len(classical_solutions) else None,
            'quantum_solution': quantum_solutions[i] if i < len(quantum_solutions) else None
        }
        case_data.append(case_info)

        # Log case info
        logger.info(f"Case {case_id}: {case_params}")
        if metadata:
            logger.info(f"  Metadata: original={metadata.get('_original_case_id')}, "
                       f"list_params={metadata.get('_list_params')}")

    # Generate the fidelity plot only if we have quantum solutions
    if quantum_solutions:
        plot_output_path = globalArgs["savedir"] + "/qlsa_fidelity_convergence.png"

        logger.info("Generating fidelity plot with automatic series detection")
        plot_path = plot_qlsa_generic(
            case_data=case_data,
            output_path=plot_output_path,
            show_plot=globalArgs.get("show_plot", False)
        )

        logger.info(f"Generated fidelity convergence plot: {plot_path}")
        if exec_status:
            lwfManager.notatePut(plot_path, exec_status.getJobContext(), {})
    else:
        logger.info("Skipping fidelity plot (no quantum solutions)")

    # Check if this is a scaling analysis (has circuit metadata)
    has_scaling_data = any('_circuit_qubits' in case['params'] for case in case_data)
    
    if has_scaling_data:
        logger.info("Detected scaling analysis data, generating scaling plots")

        scaling_plot_path = globalArgs["savedir"] + "/scaling_analysis.png"
        plot_scaling_analysis(case_data, scaling_plot_path,
                            show_plot=globalArgs.get("show_plot", False))
        logger.info(f"Generated scaling analysis plot: {scaling_plot_path}")
        if exec_status:
            lwfManager.notatePut(scaling_plot_path, exec_status.getJobContext(), {})

        scaling_table_path = globalArgs["savedir"] + "/scaling_table.png"
        plot_scaling_table(case_data, scaling_table_path)
        logger.info(f"Generated scaling table: {scaling_table_path}")
        if exec_status:
            lwfManager.notatePut(scaling_table_path, exec_status.getJobContext(), {})

    # Save case data for UQ analysis at workflow root
    import pickle
    workflow_root = str(wf.getWorkflowId())
    results_file = globalArgs["savedir"].split(workflow_root)[0] + workflow_root + "/results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(case_data, f)
    logger.info(f"Saved results for UQ analysis: {results_file}")

    # end of workflow
    logger.info(f"End of workflow {wf.getWorkflowId()}")
