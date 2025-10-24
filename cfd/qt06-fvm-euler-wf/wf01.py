"""
wf01
"""

#pylint: disable=multiple-statements. invalid-name, broad-exception-caught


import sys
from pathlib import Path
import time

from lwfm.base.Workflow import Workflow
from lwfm.midware.LwfManager import lwfManager, logger

from qtlib import get_cases_args, log_with_time, set_logger, set_workflow_start_time

from case_preproc import case_preproc

# ******************************************************************************

def run_workflow():
    """
    Execute the main workflow for all cases.
    """
    workflow_start_time = time.time()
    set_workflow_start_time(workflow_start_time)
    set_logger(logger)
    last_time = workflow_start_time
    log_with_time("Starting workflow - loading case arguments", last_time)

    # ******************************************************************************
    # get the arguments for the cases in this workflow from the TOML file

    casesArgs = get_cases_args()
    globalArgs = casesArgs["global"]

    # cases can have parameter lists - log case expansion
    case_count = sum(1 for k in casesArgs if k != "global")
    logger.info(f"Loaded {case_count} cases (after list expansion)")
    for case_id, case_args in casesArgs.items():
        if case_id != "global":
            logger.info(f"  - {case_id}: {case_args}")

    # make an lwfm workflow to bundle all the cases
    log_with_time("Creating workflow object", last_time)
    wf = Workflow("fvm euler 1d", "fvm euler 1d", globalArgs)
    if (wf := lwfManager.putWorkflow(wf)) is None: sys.exit(1)
    log_with_time(f"Registered workflow {wf.getWorkflowId()}", last_time)

    # modify the output directory name to include the workflow ID
    globalArgs["savedir"] = globalArgs["savedir"] + "/" + str(wf.getWorkflowId())
    # will be altered per case, so keep a copy of the root dir
    keepSaveDir = globalArgs["savedir"]

    # warm up lwfm sandboxes we use by updating their respective dependencies
    if globalArgs.get("warmup_sites", True):
        log_with_time("Warming up lwfm sites (updating dependencies)", last_time)
        lwfManager.updateSite()  # this projct folder ("./.venv")
        lwfManager.updateSite(globalArgs["preprocess_site"])  # make circuits
        lwfManager.updateSite(globalArgs["exec_site"])  # run circuits
        log_with_time("Site warmup complete", last_time)

    preprocess_site = lwfManager.getSite(globalArgs["preprocess_site"])
    exec_site = lwfManager.getSite(globalArgs["exec_site"])

    logger.info(f"Preprocess site: {preprocess_site.getSiteName()}")
    logger.info(f"Exec site: {exec_site.getSiteName()}")

    # ******************************************************************************
    # for each case:
    #       1. circuit generation
    #       2. circuit execution
    #       3. post processing

    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        case_start_time = time.time()
        log_with_time(f"========== Starting case: {caseId} ==========", case_start_time)

        # get the args for this case and merge in the global args
        globalArgs["savedir"] = keepSaveDir + "/" + caseId
        caseArgs.update(globalArgs)

        # put all artifacts for this case in its own subdir of workflow root
        caseOutDir = Path(globalArgs["savedir"])
        caseOutDir.mkdir(parents=True, exist_ok=True)


        # **************************************************************************
        # 1. circuit generation/preprocessing

        log_with_time(f"[{caseId}] Phase 1: Starting circuit generation/preprocessing",
            case_start_time)

        if caseArgs.get("generate_circuit", True):
            log_with_time(f"[{caseId}] Submitting circuit generation job", case_start_time)
            case_preproc(wf, caseId, caseArgs, case_start_time, preprocess_site, caseOutDir)

        log_with_time(f"[{caseId}] Circuit generation phase complete", case_start_time)


        # **************************************************************************
        # 2. circuit execution step

        log_with_time(f"[{caseId}] Phase 2: Circuit execution", case_start_time)

        # Check if we should run the circuit
        run_circuit = caseArgs.get('run_circuit', True)
        if run_circuit:
            log_with_time(f"[{caseId}] Submitting circuit execution job", case_start_time)
            # TODO: submit circuit execution job

        log_with_time(f"[{caseId}] Circuit execution phase complete", case_start_time)


        # **************************************************************************
        # 3. per-case postprocess step

        log_with_time(f"[{caseId}] Phase 3: Post-processing results", case_start_time)

        # Check if we should run the postprocess
        run_postprocess = caseArgs.get('run_postprocess', True)
        if run_postprocess:
            log_with_time(f"[{caseId}] Submitting postprocess job", case_start_time)
            # TODO: submit postprocess job

        log_with_time(f"[{caseId}] Post-processing phase complete", case_start_time)

        case_elapsed = time.time() - case_start_time
        log_with_time(f"[{caseId}] Case complete (case time: {case_elapsed:.2f}s)",
            case_start_time)

        # **************************************************************************
        # end of case loop
        # **************************************************************************

    workflow_elapsed = time.time() - workflow_start_time
    log_with_time(f"***** End of case iterations (total workflow time: {workflow_elapsed:.2f}s)",
        last_time)

    return wf


# **********************************************************************************

if __name__ == '__main__':

    # Run the main workflow and get results
    workflow = run_workflow()

    # end of workflow
    logger.info(f"***** End of workflow {workflow.getWorkflowId()}")
