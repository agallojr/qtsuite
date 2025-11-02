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
from typing import List, Any

from qtlib import log_with_time

from wf_report_generator import WorkflowReportGenerator
from wf_circuit_executor import execute_circuit

def wf_time_solver(wfId: str,
    caseId: str, caseArgs: dict, caseOutDir: Path,
    startTimes: List[float], show_plots: bool = False):
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
    if show_plots:
        args_list.append("-show_plots")

    # make sure we have a directory to put the results for this case
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

    cmd = ["python", caseArgs['solver_path']] + args_list
    log_with_time(f"[{caseId}] Running solver: {' '.join(cmd)}", startTimes)

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
        process = subprocess.Popen(
            cmd,
            stdout=None,  # Inherit parent's stdout
            stderr=None,  # Inherit parent's stderr
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create a new process group
        )
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

    # Generate workflow report if using quantum solver
    if caseArgs.get('linsolver') == 'HHL':
        print(f"[{caseId}] Generating workflow report...")
        try:
            report_gen = WorkflowReportGenerator(casedir, caseId)
            json_path, text_path = report_gen.generate_complete_report(caseArgs)
            log_with_time(f"[{caseId}] Reports generated: {json_path.name}, "
                f"{text_path.name}", startTimes)
        except Exception as e:
            print(f"[WARNING] Failed to generate report: {e}")


def wf_time_solver_circuit(execute: bool, casedir: str,
    iter_num: int, subiter_num: int = 0) -> Any:
    """
    Callback from time solver to execute circuit on quantum backend.

    Args:
        casedir: Path to case directory containing wf_context.json
        iter_num: Current outer iteration number
        subiter_num: Current inner (Newton) iteration number
    """
    # Initialize timing for this callback
    start_times: list[float] = []

    # Read context from file
    print("Reading context for circuit execution callback...")
    context_file = Path(casedir) / "wf_context.json"
    print(f"Looking for context file at {context_file}")
    if not context_file.exists():
        print(f"[ERROR] No context file at {context_file}, skipping circuit execution")
        return None

    with open(context_file, 'r', encoding='utf-8') as f:
        context = json.load(f)

    caseId = context.get('case_id', 'unknown')
    log_with_time(
        f"[{caseId}] Circuit execution callback for iter={iter_num}, subiter={subiter_num}",
        start_times
    )

    try:
        # Call execute_circuit directly with run_circuit flag
        ret = execute_circuit(casedir, iter_num, subiter_num, run_circuit=execute)
        log_with_time(f"[{caseId}] Circuit execution completed", start_times)
        return ret
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to execute circuit: {e}\n")
        return None
