"""
"""

#pylint: disable=wrong-import-position, invalid-name, superfluous-parens

# we're going to use wciscc2025 directly, but it isn't packaged as we would like,
# so manually put it on the path
import sys
sys.path.append("wciscc2025/qlsa")

import tomllib
from pathlib import Path

from lwfm.base.Workflow import Workflow
from lwfm.base.JobDefn import JobDefn
from lwfm.midware.LwfManager import lwfManager, logger
from lwfm.base.JobStatus import JobStatus

if __name__ == '__main__':
    # load the TOML file of test case inputs
    try:
        with open("./wf01-in.toml", "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        print("Error: wf01-in.toml not found.")
        sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        print(f"Error decoding TOML: {e}")
        sys.exit(1)
    globalArgs = data["global"]

    # make a workflow to bundle all these cases
    wf = Workflow("HHL size variations",
        "Fixing most params, varying matrix size - grid resolution & qubits - on various backends",
        {})    # TODO: what workflow-level metadata is useful here?
    wf = lwfManager.putWorkflow(wf)
    logger.setContext(wf)
    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())

    # for each case in the workflow toml
    for caseName, caseArgs in ((k, v) for k, v in data.items() if k != "global"):
        # get the args for this case and merge in the global args
        caseArgs.update(globalArgs)
        caseArgs["case_name"] = caseName

        # take the templatized ORNL input_vars.yaml, fill it in with the case args, save it
        with open("./input_vars.yaml", "r", encoding="utf-8") as f:
            input_vars = f.read()
        for key, value in caseArgs.items():
            input_vars = input_vars.replace("$" + key, str(value))
        out_dir = Path(globalArgs["savedir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir.joinpath(f"input_vars_{caseName}.yaml")
        circuit_qpy_path = out_dir.joinpath(f"{caseName}_circ_nqmatrix{caseArgs['NQ_MATRIX']}.qpy")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(input_vars)

        # various parts to each case - circuit generation/preprocessing, execution, post processing
        # case preprocess step
        # TODO vary the 'sample-tridiag'
        preprocess_site = lwfManager.getSite("local")
        preprocess_status = preprocess_site.getRunDriver().submit(
            JobDefn("python wciscc2025/qlsa/circuit_HHL.py", JobDefn.ENTRY_TYPE_SHELL,
                ["-case", "sample-tridiag", "-casefile", str(out_path), "--savedata"]),
                wf
        )
        if (preprocess_status is None):
            logger.error(f"Preprocess job submission failed {caseName}")
            continue
        preprocess_status = lwfManager.wait(preprocess_status.getJobId())
        if (preprocess_status is None) or (preprocess_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Preprocess job failed {caseName}")
            continue

        # case execution step
        exec_site = lwfManager.getSite("ibm-quantum-venv")
        computeType = "statevector_sim_aer"     # TODO make this an arg to the workflow / case
        runArgs = {"shots": 1024}               # TODO make this an arg to the workflow / case
        jobDefn = JobDefn(str(circuit_qpy_path), JobDefn.ENTRY_TYPE_STRING, {"format": "qpy"})
        exec_status = exec_site.getRunDriver().submit(jobDefn, wf, computeType, runArgs)
        if exec_status is None:
            logger.error(f"Circuit execution job submission failed {caseName}")
            continue
        exec_status = lwfManager.wait(exec_status.getJobId())    # TODO make async
        if (exec_status is None) or (exec_status.getStatus() != JobStatus.COMPLETE):
            logger.error(f"Circuit execution job failed {caseName}")
            continue

        # case postprocess step
        result = lwfManager.deserialize(exec_status.getNativeInfo())  # type: ignore
        logger.info(f"Circuit execution job completed {caseName}: {result}")

    # workflow postprocess step
    # TODO
