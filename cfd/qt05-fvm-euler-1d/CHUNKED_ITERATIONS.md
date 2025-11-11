# Chunked Iterations Feature

## Overview
Added support for chunked iterations with checkpoint/resume functionality to the workflow system. When enabled, the workflow runs a single iteration and then prints the command to run the next iteration. You manually drive the iteration loop from outside.

## TOML Configuration

Add this parameter to your TOML input file under `[global]`:

```toml
[global]
# ... existing parameters ...

# Chunked iterations settings
chunked_iters = true        # Enable chunked execution (default: false)
                            # Always runs 1 iteration per chunk
```

## How It Works

### When `chunked_iters = false` (default)
- Solver runs once with all iterations
- Standard behavior, no checkpointing

### When `chunked_iters = true`
1. **First run**: `python wf.py input/your-file.toml` - Creates new workflow, runs 1 iteration with `-checkpoint`
2. **Prints next command**: Displays the command to run the next iteration (does NOT execute it)
3. **You manually run**: Execute `python wf.py --resume-workflow input/your-file.toml`
4. **Subsequent runs**: Uses existing workflow ID, solver automatically detects checkpoint and runs with `-resume` flag
5. **Repeat** until solution converges or max iterations reached

### Convergence Detection

The workflow detects convergence by:
1. Checking solver exit code (must be 0)
2. Reading the latest checkpoint file (`checkpoint_iter*.pkl`)
3. Checking if `checkpoint['converged'] == True`

### Command Output

After each iteration, the workflow prints:
```
================================================================================
NEXT ITERATION COMMAND:
================================================================================
python wf.py --resume-workflow input/03-chunked-in.toml

Or run solver directly:
cd /path/to/case/output/dir
python /path/to/nozzle_1d_solver.py -nelem 2 -checkpoint -resume ...
================================================================================
```

When converged:
```
================================================================================
[case0] SOLUTION CONVERGED
================================================================================
```

## Example Usage

```bash
# First iteration - creates new workflow
python wf.py input/03-chunked-in.toml
# ... creates workflow abc12345, runs iteration 0, prints next command ...

# Second iteration - resumes existing workflow (copy/paste the printed command)
python wf.py --resume-workflow input/03-chunked-in.toml
# ... resumes workflow abc12345, runs iteration 1, prints next command ...

# Continue until converged
python wf.py --resume-workflow input/03-chunked-in.toml
# ... eventually prints "SOLUTION CONVERGED" ...
```

## Files Modified

- **wf.py**: Added workflow resume functionality
  - `--resume-workflow` / `-r` flag: Resume existing workflow instead of creating new one
  - `find_latest_workflow()`: Finds most recent workflow ID in savedir
  
- **wf_time_solver.py**: Added chunked iteration logic
  - `_check_convergence()`: Checks checkpoint for convergence
  - `_run_single_iteration_and_print_next()`: Runs 1 iteration and prints next command
  - `_run_solver_once()`: Executes single solver run
  
- **input/03-chunked-in.toml**: Example configuration with chunked iterations

## Checkpoint Files

Checkpoints are saved in the case output directory:
- `checkpoint_iter0.pkl`, `checkpoint_iter1.pkl`, etc.
- Each contains solver state for resume
- Final checkpoint marked with `converged=True` when solution converges

## Notes

- **First run**: Use `python wf.py input/file.toml` to create a new workflow
- **Subsequent runs**: Use `python wf.py --resume-workflow input/file.toml` to continue
- The `--resume-workflow` flag finds the most recent workflow ID automatically
- Log file is appended to across runs (mode='a')
- Each run executes exactly 1 iteration
- The `-iters` argument is automatically set to 1
- Checkpoint detection is automatic - if checkpoints exist, `-resume` is added to solver
- You control the iteration loop by manually re-running the printed command
- Total iterations limited by `max_iters` in TOML (solver will stop at this limit)
