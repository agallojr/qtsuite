"""
case preprocessing
"""

#pylint: disable=invalid-name, subprocess-popen-preexec-fn, broad-exception-caught

import json
import sys
import os
import signal
from pathlib import Path
import subprocess
from typing import List

from qtlib import log_with_time

def wf_time_solver(wfId: str,
    caseId: str, caseArgs: dict, caseOutDir: Path,
    startTimes: List[float]):
    """
    called from the main workflow script
    runs a single case's processing step - build up a command line and submit it to the
    preprocess site, then wait on it to complete
    """
    # Build argument list for nozzle_1d_solver.py
    args_list = []
    if 'nelem' in caseArgs:
        args_list.extend(["-nelem", str(caseArgs['nelem'])])
    if 'scheme' in caseArgs:
        args_list.extend(["-time_scheme", caseArgs['scheme']])
    if 'linsolver' in caseArgs:
        args_list.extend(["-linsolver", caseArgs['linsolver']])
    if 'cfl' in caseArgs:
        args_list.extend(["-cfl", str(caseArgs['cfl'])])
    if 'max_iters' in caseArgs:
        args_list.extend(["-iters", str(caseArgs['max_iters'])])
    if 'max_inner_iters' in caseArgs:
        args_list.extend(["-initers", str(caseArgs['max_inner_iters'])])
    if caseArgs.get('localdt', False):
        args_list.append("-localdt")
    args_list.extend(["-backend", caseArgs.get('backend', 'ideal')])
    args_list.append("-savedata")
    if caseArgs.get('hideplots', False):
        args_list.append("-hideplots")


    # make sure we have a directory to put the results for this case
    print(f"**** caseOutDir: {caseOutDir}")
    caseOutDir = caseOutDir.expanduser()
    caseOutDir.mkdir(parents=True, exist_ok=True)
    casedir = caseOutDir.as_posix()
    print(f"args_list: {args_list}")

    # Write context only if using quantum solver
    if caseArgs.get('linsolver') in ['HHL']:
        context_file = caseOutDir / "wf_context.json"
        context_data = {
            "wf_id": wfId,
            "case_id": caseId,
            "casedir": casedir,
            "case_args": {
                k: v for k, v in caseArgs.items()
                if isinstance(v, (str, int, float, bool))
            },
            "start_times": startTimes
        }
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2)
        print(f"[{caseId}] Wrote workflow context for HHL execution")
        log_with_time(f"[{caseId}] Next step is the main solver. "
            "This might take a while...", startTimes)

    # Run the solver directly using subprocess
    # Resolve solver_path to absolute path since we'll run in a different cwd
    solver_path = Path(caseArgs['solver_path']).expanduser().resolve()
    cmd = ["python", str(solver_path)] + args_list
    log_with_time(f"[{caseId}] Running solver: {cmd}", startTimes)

    process = None

    def handle_sigint(signum, frame): #pylint: disable=unused-argument
        if process:
            try:
                # Send SIGINT to the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait(timeout=1)  # Give it 5 seconds to clean up
            except subprocess.TimeoutExpired:
                process.terminate()  # Force terminate if not responding to SIGINT
            except Exception as e:
                sys.stderr.write(f"Error while handling SIGINT: {e}\n")
        sys.exit(1)  # Exit with non-zero status to indicate interrupted

    # Set up the signal handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        # Open log file for capturing stdout and stderr
        log_file_path = caseOutDir / f"{caseId}_solver.log"
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=caseOutDir,  # Run in the case output directory
                preexec_fn=os.setsid  # Create a new process group
            )

            # Read and echo output line by line
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()

            process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd,
                output=None,
                stderr=None
            )

    except Exception:
        if process and process.poll() is None:  # If process is still running
            process.terminate()
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)

    log_with_time(f"[{caseId}] Solver & circuit construction complete", startTimes)
