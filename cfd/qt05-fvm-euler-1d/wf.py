"""
wf01
"""

#pylint: disable=multiple-statements. invalid-name, broad-exception-caught


import sys
import argparse
from pathlib import Path
import time
import uuid
import json
import glob

# Import local modules
from qtlib import log_with_time, get_cases_args
from wf_time_solver import wf_time_solver
from wf_mine_results import mine_results
from wf_postprocess import postprocess

# ******************************************************************************

def find_latest_workflow(savedir: str) -> str:
    """
    Find the most recent workflow ID in the savedir.
    """
    savedir_path = Path(savedir).expanduser()
    if not savedir_path.exists():
        return None
    
    # Look for workflow directories (8-char UUIDs)
    workflow_dirs = [d for d in savedir_path.iterdir() 
                     if d.is_dir() and len(d.name) == 8]
    
    if not workflow_dirs:
        return None
    
    # Return the most recently modified
    latest = max(workflow_dirs, key=lambda d: d.stat().st_mtime)
    return latest.name


def run_workflow(workflowToml: str, resume_workflow: bool = False, submit_next: bool = False):
    """
    Execute the main workflow for all cases.
    
    Args:
        workflowToml: Path to TOML configuration file
        resume_workflow: If True, resume existing workflow instead of creating new one
        submit_next: If True, submit next iteration via sbatch instead of printing command
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

    # Create or resume workflow ID
    if resume_workflow:
        # Find existing workflow
        savedir = casesArgs.get('global', {}).get('savedir', '~/.lwfm/out/fvm-euler-1d-solver')
        workflow_id = find_latest_workflow(savedir)
        if workflow_id is None:
            print("ERROR: No existing workflow found to resume")
            sys.exit(1)
        log_with_time(f"Resuming workflow {workflow_id}", startTimes)
    else:
        # Create a short UUID for the workflow ID (first 8 chars of a UUID4)
        workflow_id = str(uuid.uuid4())[:8]
        log_with_time(f"Created workflow {workflow_id}", startTimes)

    # ******************************************************************************

    startTimes.append(time.time())
    caseDirs = []
    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        startTimes[1] = time.time()
        log_with_time(f"========== Starting case: {caseId} ==========", startTimes)

        caseArgs["savedir"] = caseArgs["savedir"] + "/" + workflow_id + "/" + caseId

        # put all artifacts for this case in its own subdir of workflow root
        caseOutDir = Path(caseArgs["savedir"]).expanduser()
        caseOutDir.mkdir(parents=True, exist_ok=True)

        # write the case arguments to a file
        caseArgsFile = caseOutDir / "case_args.json"
        with open(caseArgsFile, 'w', encoding='utf-8') as f:
            json.dump(caseArgs, f, indent=2)
        log_with_time(f"[{caseId}] Wrote case arguments to {caseArgsFile}", startTimes)

        # main event - call the wrapper for the external time solver
        log_with_time(f"[{caseId}] Running time solver", startTimes)
        wf_time_solver(workflow_id, caseId, caseArgs, caseOutDir, startTimes, submit_next)
        log_with_time(f"[{caseId}] Time solver complete", startTimes)

        # time solver writes to stdout... need to mine it for the info of interest
        mine_results(caseOutDir)
        caseDirs.append(caseOutDir)

        case_elapsed = time.time() - startTimes[1]
        log_with_time(f"[{caseId}] Case complete (case time: {case_elapsed:.2f}s)",
            startTimes)

    # end of workflow
    # ******************************************************************************

    # final post processing
    postprocess(caseDirs)

    workflow_elapsed = time.time() - startTimes[0]
    log_with_time(f"***** End of cases (total workflow time: {workflow_elapsed:.2f}s)",
        startTimes)

# **********************************************************************************

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run FVM Euler 1D workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Start new workflow
  python wf.py input/03-chunked-in.toml
  
  # Resume existing workflow (for chunked iterations)
  python wf.py --resume-workflow input/03-chunked-in.toml
        """)
    
    parser.add_argument('config_file', help='TOML configuration file')
    parser.add_argument('--resume-workflow', '-r', action='store_true',
                       help='Resume existing workflow instead of creating new one')
    parser.add_argument('--submit-next', action='store_true',
                       help='Submit next iteration via sbatch (for HPC scheduler mode)')
    
    args = parser.parse_args()
    
    # Run the main workflow and get results
    run_workflow(args.config_file, resume_workflow=args.resume_workflow, 
                 submit_next=args.submit_next)
