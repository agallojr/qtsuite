"""
"""

#pylint: disable=wrong-import-position, invalid-name, superfluous-parens

# we're going to use wciscc2025 directly, but it isn't packaged as we would like,
# so manually put it on the path
import sys
sys.path.append("wciscc2025/qlsa")

import tomllib
import argparse
from pathlib import Path

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager, logger
from lwfm.base.JobStatus import JobStatus

if __name__ == '__main__':
    # parse CLI args for workflow input file (required)
    parser = argparse.ArgumentParser(description="Run HHL workflow cases from a TOML definition")
    parser.add_argument("workflow_toml", metavar="WORKFLOW_TOML", help="Path to wf01-in TOML file")
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

    # make a workflow to bundle all these cases
    wf = Workflow("HHL size variations",
        "Fixing most params, varying matrix size - grid resolution & qubits - on various backends",
        {})    # TODO: what workflow-level metadata is useful here?
    wf = lwfManager.putWorkflow(wf)
    if wf is None:
        print("Error: Failed to register workflow.")
        sys.exit(1)
    logger.setContext(wf)
    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())

    # warm up sandboxes we use
    lwfManager.updateSite("ibm-quantum-venv")

    keepSaveDir = globalArgs["savedir"]
    # for each case in the workflow toml
    for caseId, caseArgs in ((k, v) for k, v in data.items() if k != "global"):
        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId
        caseArgs.update(globalArgs)

        # let's put all the artifacts for this case in its own directory, especially
        # because the ORNL code decides the output file names and they might otherwise clobber
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
        # (doc index 1) for 'hele-shaw', write a two-document YAML file where
        # the first document is a minimal placeholder and the second is the
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

        # various parts to each case - circuit generation/preprocessing, execution, post processing

        # in circuit generation, we need to discretize the governing Hele-Shaw equations into 
        # their Ax=B linear form. the Jacobian matrix A represents the equations, and vector b
        # the boundary conditions. all of this is done for us by the circuit_HHL.py script given 
        # parameters found in a casefile - in goes things like grid resolution, number qubits, etc.
        # and out comes a quantum circuit in a Qiskit-portable QPY format.

        preprocess_site = lwfManager.getSite("local")
        preprocess_status = preprocess_site.getRunDriver().submit(
            JobDefn("python wciscc2025/qlsa/circuit_HHL.py", JobDefn.ENTRY_TYPE_SHELL,
                ["-case", caseArgs['case'], "-casefile", str(out_path), "--savedata"]),
                wf
        )
        if (preprocess_status is None):
            logger.error(f"Preprocess job submission failed {caseId}")
            continue
        preprocess_status = lwfManager.wait(preprocess_status.getJobId())
        if (preprocess_status is None) or (preprocess_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Preprocess job failed {caseId}")
            continue

        # locate the QPY file produced by the preprocess step. The wciscc2025
        # code composes the filename based on the actual matrix size (may pad
        # up to a power of two), so don't assume NQ_MATRIX from the TOML.
        qpy_candidates = list(caseOutDir.glob(f"{caseArgs['case']}_circ_nqmatrix*.qpy"))
        if not qpy_candidates:
            logger.error(f"No generated .qpy found for {caseId} in {caseOutDir}")
            continue
        # pick the most recently modified QPY in case multiple exist
        circuit_qpy_path = max(qpy_candidates, key=lambda p: p.stat().st_mtime)

        # case execution step - use a venv site with the latest Qiskit libs
        exec_site = lwfManager.getSite("ibm-quantum-venv")
        computeType = "statevector_sim_aer"     # TODO make this an arg to the workflow / case
        runArgs = {"shots": 1024}               # TODO make this an arg to the workflow / case
        logger.info(f"Submitting job for case {caseId} with circuit {circuit_qpy_path}")
        jobDefn = JobDefn(str(circuit_qpy_path), JobDefn.ENTRY_TYPE_STRING, {"format": "qpy"})
        exec_status = exec_site.getRunDriver().submit(jobDefn, wf, computeType, runArgs)
        if exec_status is None:
            logger.error(f"Circuit execution job submission failed {caseId}")
            continue
        exec_status = lwfManager.wait(exec_status.getJobId())    # TODO make async
        if (exec_status is None) or (exec_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Circuit execution job failed {caseId}")
            continue

        # case postprocess step
        result = lwfManager.deserialize(exec_status.getNativeInfo())  # type: ignore
        logger.info(f"Circuit execution job completed {caseId}: {result}")

    # workflow postprocess step
    # TODO
