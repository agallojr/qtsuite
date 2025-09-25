"""
wf01

A workflow to create quantum circuits from ORNL's Hele-Shaw solver and run them on IBM's quantum
computers. We'll use lwfm to manage the workflow, track the artifacts, and keep different sets of
dependent libs separated.

1. Read in a TOML file with a list of cases to run
2. For each case, 
    a. Submit a job to generate the quantum circuit
    b. Submit a job to run the quantum circuit
    c. Submit a job to postprocess the results

"""

#pylint: disable=wrong-import-position, invalid-name, superfluous-parens, multiple-statements

import sys
from pathlib import Path

import qiskit    # pylint: disable=unused-import

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager, logger
from lwfm.base.JobStatus import JobStatus

from get_wf_args import get_cases_args

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
    lwfManager.updateSite("frontier-qlsa")      # makes the circuits
    lwfManager.updateSite("ibm-quantum-venv")   # runs the circuits

    # ******************************************************************************

    # for each case in the workflow toml
    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId
        caseArgs.update(globalArgs)

        # we'll put all the artifacts for this case in its own subdir of the workflow root
        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)

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
        lwfManager.notatePut(out_path.as_posix(), wf.getWorkflowId(), {"case": caseId})

        # various parts to each case - circuit generation/preprocessing, execution, post processing

        # in circuit generation, we need to discretize the governing Hele-Shaw equations into
        # their Ax=B linear form. the Jacobian matrix A represents the equations, and vector b
        # the boundary conditions. all of this is done for us by the circuit_HHL.py script given
        # parameters found in a casefile - in goes things like grid resolution, number qubits, etc.
        # and out comes a quantum circuit in a Qiskit-portable QPY format.
        preprocess_site = lwfManager.getSite("frontier-qlsa")
        preprocess_status = preprocess_site.getRunDriver().submit(
            JobDefn(f"python {caseArgs['circuit_hhl_path']}", JobDefn.ENTRY_TYPE_SHELL,
                ["-case", caseArgs['case'], "-casefile", str(out_path), "--savedata"]), wf)
        if (preprocess_status is None):
            logger.error(f"Preprocess job submission failed {caseId}")
            continue  # to next case
        preprocess_status = lwfManager.wait(preprocess_status.getJobId())
        if (preprocess_status is None) or (preprocess_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Preprocess job failed {caseId}")
            continue  # to next case

        # locate the QPY file produced by the preprocess step. The wciscc2025
        # code composes the filename based on the actual matrix size (may pad
        # up to a power of two), so don't assume NQ_MATRIX from the TOML.
        qpy_candidates = list(caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.qpy"))
        if not qpy_candidates:
            logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
            continue  # to next case
        # pick the most recently modified QPY in case multiple exist
        circuit_qpy_path = max(qpy_candidates, key=lambda p: p.stat().st_mtime)
        lwfManager.notatePut(circuit_qpy_path.as_posix(), wf.getWorkflowId(), {"case": caseId})

        # case execution step - use a venv site with the latest Qiskit libs
        exec_site = lwfManager.getSite("ibm-quantum-venv")
        computeType = caseArgs["qc_backend"]
        runArgs = {"shots": caseArgs["qc_shots"]}
        logger.info(f"Submitting job for case {caseId} with circuit {circuit_qpy_path}")
        jobDefn = JobDefn(str(circuit_qpy_path), JobDefn.ENTRY_TYPE_STRING, {"format": "qpy"})
        exec_status = exec_site.getRunDriver().submit(jobDefn, wf, computeType, runArgs)
        if exec_status is None:
            logger.error(f"Circuit execution job submission failed {caseId}")
            continue    # to next case
        exec_status = lwfManager.wait(exec_status.getJobId())
        if (exec_status is None) or (exec_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Circuit execution job failed {caseId}")
            continue    # to next case

        # case postprocess step
        result = lwfManager.deserialize(exec_status.getNativeInfo())  # type: ignore
        logger.info(f"Circuit execution job completed {caseId}: {result}")
        # write result to file in case directory
        result_path = caseOutDir / "results.out"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(str(result))
        lwfManager.notatePut(result_path.as_posix(), wf.getWorkflowId(), {"case": caseId})

        # end of case loop

    # end of workflow
    logger.info(f"End of workflow {wf.getWorkflowId()}")
