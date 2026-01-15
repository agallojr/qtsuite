#!/usr/bin/env python3
"""
Smoke test script for all quantum experiment scripts.

Runs each experiment script with minimal parameters to verify:
1. Script executes without errors
2. Valid JSON output is produced
3. Standardized fields (algorithm, script_name) are present
4. Library versions are captured

Usage:
    python smoke_test.py              # Run all tests (direct execution)
    python smoke_test.py 0            # Run mod0 tests (direct execution)
    python smoke_test.py 0s           # Run mod0 tests (via sweeper)
    python smoke_test.py 1            # Run mod1 tests (direct execution)
    python smoke_test.py 0 1          # Run mod0 and mod1 tests (direct)
    python smoke_test.py 0s 1s        # Run mod0 and mod1 tests (via sweeper)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional


# ANSI color codes
class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def colorize(text: str, color: str) -> str:
    """Add color to text if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


# Define smoke test cases for each script
SMOKE_TESTS = [
    # mod0/src/
    {
        "script": "mod0/src/hello_world_bell.py",
        "args": [],
        "algorithm": None,  # Simple script, no standardized output
        "timeout": 30,
        "skip_validation": True
    },
    {
        "script": "mod0/src/qubits.py",
        "args": ["prep", "-n", "2", "--no-display"],
        "algorithm": None,  # Simple script, no standardized output
        "timeout": 30,
        "skip_validation": True
    },
    {
        "script": "mod0/src/qubits.py",
        "args": ["hadamard", "-n", "3", "--no-display"],
        "algorithm": None,  # Simple script, no standardized output
        "timeout": 30,
        "skip_validation": True
    },
    {
        "script": "mod0/src/qubits.py",
        "args": ["ghz", "-n", "3", "--no-display"],
        "algorithm": None,  # Simple script, no standardized output
        "timeout": 30,
        "skip_validation": True
    },
    {
        "script": "mod0/src/qubits.py",
        "args": ["amplitude", "--data", "0.5", "0.5", "0.5", "0.5", "--no-display"],
        "algorithm": None,  # Simple script, no standardized output
        "timeout": 30,
        "skip_validation": True
    },
    
    # mod1/src/
    {
        "script": "mod1/src/gs_qpe.py",
        "args": ["--molecule", "H2", "--bond-length", "0.74", "--shots", "1024"],
        "algorithm": "qpe",
        "timeout": 60
    },
    {
        "script": "mod1/src/gs_vqe.py",
        "args": ["--molecule", "H2", "--shots", "1024", "--method", "qiskit"],
        "algorithm": "vqe",
        "timeout": 60
    },
    {
        "script": "mod1/src/image_flip.py",
        "args": ["--size", "4", "--shots", "512"],
        "algorithm": "image_flip",
        "timeout": 30
    },
    {
        "script": "mod1/src/phase_kickback.py",
        "args": [],
        "algorithm": "phase_kickback",
        "timeout": 30
    },
    {
        "script": "mod1/src/lattice_vqe.py",
        "args": ["--rows", "2", "--cols", "2", "--method", "qiskit", "--maxiter", "10"],
        "algorithm": "lattice_vqe",
        "timeout": 60
    },
    {
        "script": "mod1/src/ax_equals_b_vlqs.py",
        "args": ["--size", "2", "--maxiter", "10"],
        "algorithm": "vlqs",
        "timeout": 60
    },
    
    # mod2/src/
    {
        "script": "mod2/src/ax_equals_b_cks_qrisp_est.py",
        "args": ["--matrix", "[[2,1],[1,2]]", "--vector", "[1,0]", "--epsilon", "0.1", "--shots", "512"],
        "algorithm": "cks",
        "timeout": 60
    },
    {
        "script": "mod2/src/grovers.py",
        "args": ["--targets", "101", "--n", "3", "--shots", "512"],
        "algorithm": "grover",
        "timeout": 30
    },
    {
        "script": "mod2/src/ax_equals_b_hhl.py",
        "args": ["--matrix", "[[2,1],[1,2]]", "--vector", "[1,0]", "--shots", "512"],
        "algorithm": "hhl",
        "timeout": 60
    },
    {
        "script": "mod2/src/ax_equals_b_hhl_qrisp.py",
        "args": ["--matrix", "[[2,1],[1,2]]", "--vector", "[1,0]", "--precision", "2", "--trotter-steps", "2", "--shots", "512"],
        "algorithm": "hhl",
        "timeout": 60
    },
    
    # mod3/src/
    # Note: est_wannier.py requires JARVIS API access (skipped for smoke test)
    # Note: gs_sqd.py has numerical issues with qiskit_addon_sqd library (skipped for smoke test)
    {
        "script": "mod3/src/gs_siam_skqd.py",
        "args": ["--num-orbs", "4", "--krylov-dim", "2", "--max-iter", "1", 
                 "--num-batches", "2", "--samples-per-batch", "50", "--shots", "512"],
        "algorithm": "sqd",
        "timeout": 120
    }
]


def run_smoke_test(test: Dict, qp4p_root: Path, python_path: str) -> Tuple[bool, str, Dict]:
    """
    Run a single smoke test.
    
    Returns:
        (success, message, output_json)
    """
    script_path = qp4p_root / test["script"]
    script_name = Path(test["script"]).name
    skip_validation = test.get("skip_validation", False)
    
    if not script_path.exists():
        return False, f"Script not found: {script_path}", {}
    
    # Build command
    cmd = [python_path, str(script_path)] + test["args"]
    
    try:
        # Run the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test["timeout"],
            cwd=qp4p_root
        )
        
        # Check return code
        if result.returncode != 0:
            return False, f"Non-zero exit code: {result.returncode}\nStderr: {result.stderr}", {}
        
        # For simple scripts that don't use standardized output, just check they ran
        if skip_validation:
            return True, "OK (no validation)", {}
        
        # Parse JSON output
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON output: {e}\nStdout: {result.stdout[:500]}", {}
        
        # Validate standardized fields
        if "algorithm" not in output:
            return False, "Missing 'algorithm' field in output", output
        
        if "script_name" not in output:
            return False, "Missing 'script_name' field in output", output
        
        if output["algorithm"] != test["algorithm"]:
            return False, f"Algorithm mismatch: expected '{test['algorithm']}', got '{output['algorithm']}'", output
        
        if output["script_name"] != script_name:
            return False, f"Script name mismatch: expected '{script_name}', got '{output['script_name']}'", output
        
        if "library_versions" not in output:
            return False, "Missing 'library_versions' field in output", output
        
        if "status" in output and output["status"] == "error":
            return False, f"Script reported error status: {output.get('error', 'Unknown error')}", output
        
        # Success
        return True, "OK", output
        
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {test['timeout']} seconds", {}
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}", {}


def filter_tests_by_modules(tests: List[Dict], modules: Optional[List[int]]) -> List[Dict]:
    """Filter tests to only include specified modules."""
    if modules is None:
        return tests
    
    filtered = []
    for test in tests:
        script_path = test["script"]
        # Extract module number from path like "mod0/src/..."
        if script_path.startswith("mod"):
            mod_num = int(script_path[3])  # Get digit after "mod"
            if mod_num in modules:
                filtered.append(test)
    
    return filtered


def run_sweeper_tests(modules: List[int], qp4p_root: Path, python_path: str) -> None:
    """
    Run sweeper configurations for specified modules.
    
    Args:
        modules: List of module numbers to run sweeper for
        qp4p_root: Root directory of qp4p
        python_path: Path to Python interpreter
    """
    sweeper_script = qp4p_root / "helpers" / "src" / "qp4p_sweeper.py"
    
    # Map TOML files to their corresponding scripts
    toml_to_script = {
        "hello_world_bell.toml": "mod0/src/hello_world_bell.py",
        "ax_equals_b.toml": "mod1/src/ax_equals_b_vlqs.py",
        "gs_qpe.toml": "mod1/src/gs_qpe.py",
        "gs_vqe.toml": "mod1/src/gs_vqe.py",
        "image_flip.toml": "mod1/src/image_flip.py",
        "lattice_vqe.toml": "mod1/src/lattice_vqe.py",
        "phase_kickback.toml": "mod1/src/phase_kickback.py",
        "ax_equals_b_hhl.toml": "mod2/src/ax_equals_b_hhl.py",
        "ax_equals_b_hhl_qrisp.toml": "mod2/src/ax_equals_b_hhl_qrisp.py",
        "grovers.toml": "mod2/src/grovers.py",
        "est_wannier.toml": "mod3/src/est_wannier.py",
        "gs_siam_skqd.toml": "mod3/src/gs_siam_skqd.py",
        "gs_sqd.toml": "mod3/src/gs_sqd.py",
    }
    
    print(colorize("QP4P Sweeper Test Suite", Colors.BOLD + Colors.CYAN))
    print(f"QP4P Root: {qp4p_root}")
    print(f"Python: {python_path}")
    print(f"Modules: {colorize(', '.join([f'mod{m}' for m in sorted(modules)]), Colors.YELLOW)}")
    print()
    
    passed = 0
    failed = 0
    total_tests = 0
    
    for module_num in sorted(modules):
        input_dir = qp4p_root / f"mod{module_num}" / "input"
        
        if not input_dir.exists():
            print(f"{colorize('✗', Colors.RED)} Input directory not found: {input_dir}")
            continue
        
        # Find all TOML files in the input directory
        toml_files = sorted(input_dir.glob("*.toml"))
        
        if not toml_files:
            print(f"{colorize('⊘', Colors.YELLOW)} No TOML files found in {input_dir}")
            continue
        
        print(f"\n{colorize(f'Module {module_num}', Colors.BOLD)} - {len(toml_files)} sweeper config(s)")
        
        for toml_file in toml_files:
            total_tests += 1
            
            # Get the script path for this TOML file
            script_path = toml_to_script.get(toml_file.name)
            if not script_path:
                print(f"  [{total_tests}] {colorize(toml_file.name, Colors.BLUE)}... {colorize('⊘ SKIP', Colors.YELLOW)} (no script mapping)")
                continue
            
            script_full_path = qp4p_root / script_path
            if not script_full_path.exists():
                print(f"  [{total_tests}] {colorize(toml_file.name, Colors.BLUE)}... {colorize('✗ FAIL', Colors.RED)} (script not found: {script_path})")
                failed += 1
                continue
            
            print(f"  [{total_tests}] {colorize(toml_file.name, Colors.BLUE)}...", end=" ")
            
            try:
                result = subprocess.run(
                    [python_path, str(sweeper_script), str(toml_file), str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for sweeper
                    cwd=qp4p_root
                )
                
                if result.returncode == 0:
                    print(colorize('✓ PASS', Colors.GREEN + Colors.BOLD))
                    passed += 1
                else:
                    print(colorize('✗ FAIL', Colors.RED + Colors.BOLD))
                    print(f"    {colorize('Error:', Colors.RED)} Exit code {result.returncode}")
                    if result.stderr:
                        print(f"    Stderr: {result.stderr[:200]}")
                    failed += 1
            except subprocess.TimeoutExpired:
                print(colorize('✗ FAIL', Colors.RED + Colors.BOLD))
                print(f"    {colorize('Error:', Colors.RED)} Timeout after 600 seconds")
                failed += 1
            except Exception as e:
                print(colorize('✗ FAIL', Colors.RED + Colors.BOLD))
                print(f"    {colorize('Error:', Colors.RED)} {e}")
                failed += 1
    
    # Summary
    print("\n" + colorize("="*70, Colors.BOLD))
    summary = f"Sweeper Test Summary: {colorize(str(passed) + ' passed', Colors.GREEN)}, {colorize(str(failed) + ' failed', Colors.RED if failed > 0 else Colors.GREEN)} out of {total_tests} sweeps"
    print(summary)
    print(colorize("="*70, Colors.BOLD))
    
    sys.exit(0 if failed == 0 else 1)


def parse_module_arg(arg: str) -> Tuple[int, bool]:
    """
    Parse module argument, accepting both '0' and '0s' formats.
    
    Returns:
        Tuple of (module_num, use_sweeper)
        - '0' returns (0, False) - direct execution
        - '0s' returns (0, True) - sweeper execution
    """
    arg_original = arg.strip()
    arg = arg_original.lower()
    use_sweeper = arg.endswith('s')
    
    if use_sweeper:
        arg = arg[:-1]
    
    try:
        module_num = int(arg)
        if module_num not in [0, 1, 2, 3]:
            raise ValueError(f"Module number must be 0-3, got {module_num}")
        return (module_num, use_sweeper)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid module argument '{arg_original}': {e}")


def main():
    """Run all smoke tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run smoke tests for quantum experiment scripts",
        epilog="""Examples:
  %(prog)s              # Run all tests (direct execution)
  %(prog)s 0            # Run mod0 tests (direct execution)
  %(prog)s 0s           # Run mod0 tests (via sweeper)
  %(prog)s 1            # Run mod1 tests (direct execution)
  %(prog)s 0 1          # Run mod0 and mod1 tests (direct)
  %(prog)s 0s 1s        # Run mod0 and mod1 tests (via sweeper)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "modules",
        nargs="*",
        type=parse_module_arg,
        help="Module numbers to test (0-3 or 0s-3s). If not specified, all modules are tested."
    )
    args = parser.parse_args()
    
    # Find qp4p root
    script_dir = Path(__file__).parent
    qp4p_root = script_dir.parent.parent
    
    # Use venv python if available
    venv_python = qp4p_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_path = str(venv_python)
    else:
        python_path = sys.executable
    
    # Check if any modules request sweeper execution
    if args.modules:
        sweeper_modules = [mod_num for mod_num, use_sweeper in args.modules if use_sweeper]
        if sweeper_modules:
            # Run sweeper for requested modules
            return run_sweeper_tests(sweeper_modules, qp4p_root, python_path)
    
    # Filter tests by module if specified (direct execution only)
    modules = [mod_num for mod_num, use_sweeper in args.modules] if args.modules else None
    tests_to_run = filter_tests_by_modules(SMOKE_TESTS, modules)
    
    if not tests_to_run:
        print(colorize("No tests found for specified modules", Colors.RED))
        sys.exit(1)
    
    # Print header
    print(colorize("QP4P Smoke Test Suite", Colors.BOLD + Colors.CYAN))
    print(f"QP4P Root: {qp4p_root}")
    print(f"Python: {python_path}")
    if modules:
        module_str = ", ".join([f"mod{m}" for m in sorted(modules)])
        print(f"Modules: {colorize(module_str, Colors.YELLOW)}")
    else:
        print(f"Modules: {colorize('all', Colors.YELLOW)}")
    print(f"Running {colorize(str(len(tests_to_run)), Colors.BOLD)} smoke tests...\n")
    
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests_to_run, 1):
        script_name = test["script"]
        print(f"[{colorize(f'{i}/{len(tests_to_run)}', Colors.CYAN)}] Testing {colorize(script_name, Colors.BLUE)}...", end=" ", flush=True)
        
        success, message, output = run_smoke_test(test, qp4p_root, python_path)
        
        if success:
            print(colorize("✓ PASS", Colors.GREEN + Colors.BOLD))
            passed += 1
        else:
            print(colorize("✗ FAIL", Colors.RED + Colors.BOLD))
            print(f"    {colorize('Error:', Colors.RED)} {message}")
            failed += 1
        
        results.append({
            "script": script_name,
            "success": success,
            "message": message,
            "algorithm": test["algorithm"]
        })
    
    # Summary
    print("\n" + colorize("="*70, Colors.BOLD))
    summary = f"Smoke Test Summary: {colorize(str(passed) + ' passed', Colors.GREEN)}, {colorize(str(failed) + ' failed', Colors.RED if failed > 0 else Colors.GREEN)} out of {len(tests_to_run)} tests"
    print(summary)
    print(colorize("="*70, Colors.BOLD))
    
    if failed > 0:
        print(f"\n{colorize('Failed tests:', Colors.RED + Colors.BOLD)}")
        for result in results:
            if not result["success"]:
                print(f"  {colorize('✗', Colors.RED)} {result['script']}")
                print(f"    {result['message']}")
        sys.exit(1)
    else:
        print(f"\n{colorize('✓ All smoke tests passed!', Colors.GREEN + Colors.BOLD)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
