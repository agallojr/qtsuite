"""
workflow time solver wrapper - executes nozzle_1d_solver.py with proper arguments
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
from pathlib import Path as PathlibPath
import pickle

def _get_latest_checkpoint(caseOutDir: Path):
    """Get the latest checkpoint file and data"""
    import glob
    pattern = str(caseOutDir / 'checkpoint_iter*.pkl')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, None
    
    # Find latest checkpoint
    iters = []
    for ckpt in checkpoints:
        basename = PathlibPath(ckpt).name
        iter_str = basename.replace('checkpoint_iter', '').replace('.pkl', '')
        try:
            iters.append((int(iter_str), ckpt))
        except ValueError:
            continue
    
    if not iters:
        return None, None
    
    latest_ckpt = max(iters, key=lambda x: x[0])[1]
    try:
        with open(latest_ckpt, 'rb') as f:
            ckpt_data = pickle.load(f)
        return latest_ckpt, ckpt_data
    except Exception:
        return None, None


def _check_convergence(caseOutDir: Path) -> bool:
    """Check if solution has converged by examining latest checkpoint"""
    _, ckpt_data = _get_latest_checkpoint(caseOutDir)
    if ckpt_data is None:
        return False
    return ckpt_data.get('converged', False)


def _check_max_iters_reached(caseOutDir: Path, original_max_iters: int) -> bool:
    """Check if we've reached the original max_iters from TOML"""
    _, ckpt_data = _get_latest_checkpoint(caseOutDir)
    if ckpt_data is None:
        return False
    
    # Get the iteration we just completed
    current_iter = ckpt_data.get('iteration', -1)
    # Get the original max from checkpoint (or use passed value as fallback)
    checkpoint_max = ckpt_data.get('original_max_iters', original_max_iters)
    
    # We've reached max if current_iter + 1 >= original_max_iters
    # (current_iter is 0-indexed, so iter 2 means we've done 3 iterations: 0, 1, 2)
    return (current_iter + 1) >= checkpoint_max


def _print_next_iteration_command(caseOutDir: Path, startTimes: List[float]):
    """Print the command to run the next iteration (local mode)"""
    print(f"\n{'='*80}")
    print(f"NEXT ITERATION COMMAND:")
    print(f"{'='*80}")
    # Build the command with --resume-workflow flag
    wf_args = [arg for arg in sys.argv[1:] if arg not in ['--resume-workflow', '-r', '--submit-next']]
    print(f"python wf.py --resume-workflow {' '.join(wf_args)}")
    print(f"{'='*80}\n")


def _submit_next_job(startTimes: List[float], caseId: str):
    """Submit the next iteration as a Slurm job (HPC mode)"""
    import subprocess
    
    # Get the script directory (assumes wf.py is in project root, scripts/ is sibling)
    script_dir = Path(__file__).parent / 'scripts'
    slurm_script = script_dir / 'wf.sl'
    
    # Build sbatch command with same arguments but add --resume-workflow
    wf_args = [arg for arg in sys.argv[1:] if arg not in ['--resume-workflow', '-r']]
    if '--resume-workflow' not in wf_args and '-r' not in wf_args:
        wf_args.insert(0, '--resume-workflow')
    
    sbatch_cmd = ['sbatch', str(slurm_script)] + wf_args
    
    log_with_time(f"[{caseId}] Submitting next iteration: {' '.join(sbatch_cmd)}", startTimes)
    
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        print(f"\n{'='*80}")
        print(f"NEXT ITERATION SUBMITTED TO SLURM")
        print(f"{'='*80}")
        print(f"Job ID: {job_id}")
        print(f"Command: {' '.join(sbatch_cmd)}")
        print(f"{'='*80}\n")
        log_with_time(f"[{caseId}] Submitted job: {job_id}", startTimes)
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Failed to submit next iteration")
        print(f"{'='*80}")
        print(f"Command: {' '.join(sbatch_cmd)}")
        print(f"Error: {e.stderr}")
        print(f"{'='*80}\n")
        raise


def wf_time_solver(wfId: str,
    caseId: str, caseArgs: dict, caseOutDir: Path,
    startTimes: List[float], submit_next: bool = False):
    """
    called from the main workflow script
    runs a single case's solver - builds command line arguments and executes
    nozzle_1d_solver.py, then waits for it to complete
    
    Supports chunked iterations: if chunked_iters=true, runs solver for 1 iteration,
    then prints the command to run the next iteration (does not execute it)
    
    Args:
        submit_next: If True, submit next iteration via sbatch instead of printing command
    """
    # Check if chunked iterations are enabled
    chunked_iters = caseArgs.get('chunked_iters', False)
    chunk_size = 1  # Always use 1 iteration per chunk
    
    # Build base argument list for nozzle_1d_solver.py
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
    if caseArgs.get('hhl_scale'):
        args_list.extend(["-hhl_scale", caseArgs['hhl_scale']])
    # Handle backend - split if it contains spaces (e.g., "fake fake_brisbane")
    backend = caseArgs.get('backend', 'ideal')
    backend_parts = backend.split() if isinstance(backend, str) else [backend]
    args_list.append("-backend")
    args_list.extend(backend_parts)
    shots = caseArgs.get('shots', 0)
    if shots > 0:
        args_list.extend(["-shots", str(shots)])
    args_list.append("-savedata")
    if caseArgs.get('hideplots', False):
        args_list.append("-noshow")
    if caseArgs.get('dump_matrices', True):
        args_list.extend(["-v", "2"])
    else:
        args_list.extend(["-v", "1"])
    
    # Add checkpoint flag if chunked iterations enabled
    if chunked_iters:
        args_list.append("-checkpoint")

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

    # Resolve solver_path to absolute path since we'll run in a different cwd
    solver_path = Path(caseArgs['solver_path']).expanduser().resolve()
    
    # Run solver
    if chunked_iters:
        # Run single iteration and print next command
        _run_single_iteration_and_print_next(caseId, solver_path, args_list, 
                                             caseOutDir, caseArgs, startTimes, submit_next)
    else:
        # Run all iterations at once
        cmd = ["python", str(solver_path)] + args_list
        log_with_time(f"[{caseId}] Running solver: {cmd}", startTimes)
        returncode = _run_solver_once(cmd, caseOutDir, caseId)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    log_with_time(f"[{caseId}] Solver & circuit construction complete", startTimes)


def _run_single_iteration_and_print_next(caseId: str, solver_path: Path, 
                                         base_args: list, caseOutDir: Path,
                                         caseArgs: dict, startTimes: List[float],
                                         submit_next: bool = False):
    """Run solver for 1 iteration, then print or submit command for next iteration"""
    
    # Build command for this iteration
    args_list = base_args.copy()
    
    # Save original max_iters before overriding
    original_max_iters = caseArgs.get('max_iters', 2000)
    
    # Set max_iters to 1 for this run
    for i, arg in enumerate(args_list):
        if arg == "-iters" and i + 1 < len(args_list):
            args_list[i + 1] = "1"
            break
    
    # Add original max_iters so solver can track total limit
    args_list.extend(["-original_max_iters", str(original_max_iters)])
    
    # Check if we should resume from checkpoint
    has_checkpoint = len(list(caseOutDir.glob('checkpoint_iter*.pkl'))) > 0
    if has_checkpoint:
        args_list.append("-resume")
    
    cmd = ["python", str(solver_path)] + args_list
    log_with_time(f"[{caseId}] Running iteration: {' '.join(cmd)}", startTimes)
    
    # Run this iteration
    returncode = _run_solver_once(cmd, caseOutDir, caseId)
    
    # Check for errors
    if returncode != 0:
        log_with_time(f"[{caseId}] Solver failed with exit code {returncode}", 
                     startTimes)
        raise subprocess.CalledProcessError(returncode, cmd)
    
    # Check completion status
    converged = _check_convergence(caseOutDir)
    max_iters_reached = _check_max_iters_reached(caseOutDir, caseArgs.get('max_iters'))
    
    if converged:
        print(f"\n{'='*80}")
        print(f"[{caseId}] WORKFLOW COMPLETE - SOLUTION CONVERGED")
        print(f"{'='*80}")
        print(f"Do not run another iteration.")
        print(f"{'='*80}\n")
        log_with_time(f"[{caseId}] Solution converged", startTimes)
    elif max_iters_reached:
        print(f"\n{'='*80}")
        print(f"[{caseId}] WORKFLOW COMPLETE - MAX ITERATIONS REACHED")
        print(f"{'='*80}")
        print(f"Reached maximum iterations ({caseArgs.get('max_iters')}).")
        print(f"Do not run another iteration.")
        print(f"{'='*80}\n")
        log_with_time(f"[{caseId}] Max iterations reached", startTimes)
    else:
        # More iterations needed
        if submit_next:
            # Submit next iteration via sbatch
            _submit_next_job(startTimes, caseId)
        else:
            # Print command for next iteration (local mode)
            _print_next_iteration_command(caseOutDir, startTimes)


def _run_solver_once(cmd: list, caseOutDir: Path, caseId: str) -> int:
    """Run solver command once and return exit code"""
    process = None
    returncode = 0

    def handle_sigint(signum, frame): #pylint: disable=unused-argument
        if process:
            try:
                # Send SIGINT to the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
            except Exception as e:
                sys.stderr.write(f"Error while handling SIGINT: {e}\n")
        sys.exit(1)

    # Set up the signal handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        # Open log file for capturing stdout and stderr
        log_file_path = caseOutDir / f"{caseId}_solver.log"
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=caseOutDir,
                preexec_fn=os.setsid
            )

            # Read and echo output line by line
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()

            process.wait()

        returncode = process.returncode

    except Exception:
        if process and process.poll() is None:
            process.terminate()
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)
    
    return returncode
