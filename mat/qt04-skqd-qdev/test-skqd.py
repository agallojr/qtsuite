#!/usr/bin/env python3
"""
SKQD Test Launcher - SIAM workflow.

Usage:
    python test-skqd.py [OPTIONS]
    
Options:
    --shots N          Number of shots for execution (default: 1024)
    --large            Use large 48-qubit problem (step2_large) instead of 20-qubit (step2_small)
                       Note: 48 qubits cannot run on local simulator
    --skip-execute     Skip transpilation and execution steps
    --login            Run step0 to save IBM Quantum account (skipped by default)
"""

#pylint: disable=import-outside-toplevel

import sys


def parse_args():
    """Parse command line arguments."""
    args = {
        'shots': 1024,
        'large': '--large' in sys.argv,
        'skip_execute': '--skip-execute' in sys.argv,
        'login': '--login' in sys.argv,
    }
    
    # Parse --shots N
    for i, arg in enumerate(sys.argv):
        if arg == '--shots' and i + 1 < len(sys.argv):
            args['shots'] = int(sys.argv[i + 1])
    
    return args


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SKQD Test Suite")
    print(f"  shots={args['shots']}, large={args['large']}, skip_execute={args['skip_execute']}, login={args['login']}")
    print("=" * 60)
    
    # Step 0: Save account (optional)
    if args['login']:
        print("\n--- Step 0: Save IBM Quantum Account ---")
        from step0_login import save_account
        save_account()
    else:
        print("\n--- Step 0: Skipped (use --login to save account) ---")
    
    # Step 1: SIAM Hamiltonian in momentum basis
    print("\n--- Step 1: SIAM Hamiltonian (Momentum Basis) ---")
    from step1_siam import run_step1
    run_step1()
    
    # Step 2: Build SIAM circuits
    if args['large']:
        print("\n--- Step 2: Build Large SIAM Circuits (48 qubits) ---")
        from step2_siam_large import run_step2
        circuits = run_step2()
    else:
        print("\n--- Step 2: Build SIAM Circuits (20 qubits) ---")
        from step2_siam_small import run_step2
        circuits = run_step2()
        # Add measurements for transpile/execute
        for qc in circuits:
            qc.measure_all()
    
    # Step 3: Transpile
    # Step 4: Execute
    if args['skip_execute']:
        print("\n--- Step 3/4: Skipped (--skip-execute flag) ---")
    else:
        print("\n--- Step 3: Transpile for Aer density_matrix ---")
        from step3_transpile import run_step_transpile
        transpiled = run_step_transpile(circuits)
        
        print(f"\n--- Step 4: Execute ({args['shots']} shots) ---")
        from step4_execute import run_step_execute
        _, _, counts = run_step_execute(transpiled, shots=args['shots'])
        print(f"Unique bitstrings collected: {len(counts)}")
    
    print("\n" + "=" * 60)
    print("SKQD Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
