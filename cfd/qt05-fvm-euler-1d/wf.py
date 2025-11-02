"""
wf01
"""

#pylint: disable=multiple-statements. invalid-name, broad-exception-caught


import sys
from pathlib import Path
import time
import uuid

# Import local modules
from qtlib import log_with_time, get_cases_args
from wf_time_solver import wf_time_solver

# ******************************************************************************

def run_workflow(workflowToml: str):
    """
    Execute the main workflow for all cases.
    """
    # Parse the TOML configuration file
    casesArgs = get_cases_args(workflowToml)

    startTimes = [time.time()]
    log_with_time("Starting workflow - loading case arguments", startTimes)

    # ******************************************************************************
    # get the arguments for the cases in this workflow from the TOML file

    # Log case information
    case_count = len(casesArgs)
    log_with_time(f"Loaded {case_count} cases", startTimes)
    for case_id, case_args in casesArgs.items():
        print(f"  - {case_id}: {case_args}")

    # Create a short UUID for the workflow ID (first 8 chars of a UUID4)
    workflow_id = str(uuid.uuid4())[:8]
    log_with_time(f"Created workflow {workflow_id}", startTimes)

    # ******************************************************************************

    startTimes.append(time.time())
    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        startTimes[1] = time.time()
        log_with_time(f"========== Starting case: {caseId} ==========", startTimes)

        caseArgs["savedir"] = caseArgs["savedir"] + "/" + workflow_id + "/" + caseId

        # put all artifacts for this case in its own subdir of workflow root
        caseOutDir = Path(caseArgs["savedir"]).expanduser()
        caseOutDir.mkdir(parents=True, exist_ok=True)

        log_with_time(f"[{caseId}] Starting time solver", startTimes)

        wf_time_solver(workflow_id, caseId, caseArgs, caseOutDir,
            startTimes, show_plots=caseArgs.get("show_plots", False))

        log_with_time(f"[{caseId}] Time solver complete", startTimes)

        case_elapsed = time.time() - startTimes[1]
        log_with_time(f"[{caseId}] Case complete (case time: {case_elapsed:.2f}s)",
            startTimes)

    # end of workflow
    # ******************************************************************************

    workflow_elapsed = time.time() - startTimes[0]
    log_with_time(f"***** End of cases (total workflow time: {workflow_elapsed:.2f}s)",
        startTimes)

# **********************************************************************************

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python wf.py <config_file>")
        sys.exit(1)

    # Run the main workflow and get results
    run_workflow(sys.argv[1])
