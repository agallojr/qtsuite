#!/usr/bin/env python3
"""
SKQD Test Launcher - SIAM workflow.

Usage:
    python test-skqd.py [OPTIONS]
    
Options:
    --num-orbs N       Number of orbitals (default: 10, qubits = 2*N)
    --krylov-dim N     Krylov dimension (default: 5)
    --dt-mult N        Time step multiplier (default: 1.0)
    --shots N          Number of shots for execution (default: 1024)
    --noise N          Add depolarizing noise (0.0-1.0, default: 0 = noiseless)
    --hopping N        Hopping parameter (default: 1.0)
    --onsite N         Onsite energy U (default: 5.0)
    --hybridization N  Hybridization strength (default: 1.0)
    --filling-factor N Filling factor for chemical potential (default: -0.5)
    --opt-level N      Transpiler optimization level 0-3 (default: 1)
    --max-iter N       SQD max iterations (default: 10)
    --num-batches N    SQD number of batches (default: 5)
    --samples-per-batch N  SQD samples per batch (default: 200)

Performance guidelines:
    - qubits scale as 2*num_orbs
    - noise is essential in this algorithm
    - 
"""

#pylint: disable=import-outside-toplevel, unused-variable

import json
import sys
import time


def parse_args():
    """Parse command line arguments."""
    args = {
        'num_orbs': 10,
        'krylov_dim': 5,
        'dt_mult': 1.0,
        'shots': 1024,
        'noise': 0.0,
        'hopping': 1.0,
        'onsite': 5.0,
        'hybridization': 1.0,
        'filling_factor': -0.5,
        'opt_level': 1,
        'max_iter': 10,
        'num_batches': 5,
        'samples_per_batch': 200,
    }
    
    # Parse numeric arguments
    for i, arg in enumerate(sys.argv):
        if arg == '--num-orbs' and i + 1 < len(sys.argv):
            args['num_orbs'] = int(sys.argv[i + 1])
        elif arg == '--krylov-dim' and i + 1 < len(sys.argv):
            args['krylov_dim'] = int(sys.argv[i + 1])
        elif arg == '--shots' and i + 1 < len(sys.argv):
            args['shots'] = int(sys.argv[i + 1])
        elif arg == '--dt-mult' and i + 1 < len(sys.argv):
            args['dt_mult'] = float(sys.argv[i + 1])
        elif arg == '--noise' and i + 1 < len(sys.argv):
            args['noise'] = float(sys.argv[i + 1])
        elif arg == '--hopping' and i + 1 < len(sys.argv):
            args['hopping'] = float(sys.argv[i + 1])
        elif arg == '--onsite' and i + 1 < len(sys.argv):
            args['onsite'] = float(sys.argv[i + 1])
        elif arg == '--hybridization' and i + 1 < len(sys.argv):
            args['hybridization'] = float(sys.argv[i + 1])
        elif arg == '--filling-factor' and i + 1 < len(sys.argv):
            args['filling_factor'] = float(sys.argv[i + 1])
        elif arg == '--opt-level' and i + 1 < len(sys.argv):
            args['opt_level'] = int(sys.argv[i + 1])
        elif arg == '--max-iter' and i + 1 < len(sys.argv):
            args['max_iter'] = int(sys.argv[i + 1])
        elif arg == '--num-batches' and i + 1 < len(sys.argv):
            args['num_batches'] = int(sys.argv[i + 1])
        elif arg == '--samples-per-batch' and i + 1 < len(sys.argv):
            args['samples_per_batch'] = int(sys.argv[i + 1])
    
    return args


def main():
    start_time = time.time()
    args = parse_args()
    
    num_qubits = 2 * args['num_orbs']
    print("=" * 60)
    print("SKQD Test Suite")
    print(f"  num_orbs={args['num_orbs']} ({num_qubits} qubits), "
          f"krylov_dim={args['krylov_dim']}, dt_mult={args['dt_mult']}, "
          f"shots={args['shots']}, noise={args['noise']}")
    print(f"  hopping={args['hopping']}, onsite={args['onsite']}, "
          f"hybridization={args['hybridization']}, filling_factor={args['filling_factor']}, "
          f"opt_level={args['opt_level']}")
    print(f"  max_iter={args['max_iter']}, num_batches={args['num_batches']}, "
          f"samples_per_batch={args['samples_per_batch']}")
    print("=" * 60)
      
    # Step 1: SIAM Hamiltonian in momentum basis
    print(f"\n--- Step 1: SIAM Hamiltonian ({args['num_orbs']} orbitals) ---")
    from step1_siam import run_step1
    pre_start = time.time()
    run_step1(num_orbs=args['num_orbs'])
    pre_time = time.time() - pre_start
    print(f"Hamiltonian setup in {pre_time:.2f}s")
    
    # Step 2: Build SIAM Krylov circuits
    print(f"\n--- Step 2: Build SIAM Krylov Circuits ({num_qubits} qubits) ---")
    from step2_krylov import run_step2
    build_start = time.time()
    circuits = run_step2(num_orbs=args['num_orbs'], krylov_dim=args['krylov_dim'],
        dt_multiplier=args['dt_mult'])
    build_time = time.time() - build_start
    print(f"Built {len(circuits)} circuits in {build_time:.2f}s")
    
    # Create backend & noise model
    from qiskit_aer import AerSimulator
    if args['noise'] > 0:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        # Light depolarizing noise on 2-qubit gates only
        p = args['noise']
        noise_model = NoiseModel()
        error_2q = depolarizing_error(p, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'cp'])
        backend = AerSimulator(noise_model=noise_model)
        print(f"Using noisy backend (2q depolarizing={p})")
    else:
        backend = AerSimulator(method='automatic')
    
    # Step 3: Transpile
    print(f"\n--- Step 3: Transpile for {backend} ---")
    from step3_transpile import transpile_circuits
    transpile_start = time.time()
    transpiled = transpile_circuits(circuits, backend=backend,
                                      optimization_level=args['opt_level'])
    transpile_time = time.time() - transpile_start
    
    # Circuit depth stats
    depths = [c.depth() for c in transpiled]
    avg_depth = sum(depths) / len(depths)
    print(f"Transpiled {len(transpiled)} circuits in {transpile_time:.2f}s")
    print(f"Circuit depths: min={min(depths)}, max={max(depths)}, avg={avg_depth:.1f}")
    
    # Step 4: Execute
    print(f"\n--- Step 4: Execute ({args['shots']} shots) ---")
    from step4_execute import execute_circuits
    exec_start = time.time()
    bitstrings, probabilities, counts = execute_circuits(transpiled, backend=backend,
        shots=args['shots'])
    exec_time = time.time() - exec_start
    print(f"Executed in {exec_time:.2f}s, unique bitstrings: {len(counts)}")
    
    # Step 5: Post-process with SQD
    print("\n--- Step 5: SQD Post-processing ---")
    from step5_postprocess import run_step5, exact_siam_energy
    import skqd_helpers
    
    postprocess_start = time.time()
    result = run_step5(counts, num_orbs=args['num_orbs'],
                        hopping=args['hopping'], onsite=args['onsite'],
                        hybridization=args['hybridization'],
                        filling_factor=args['filling_factor'],
                        max_iterations=args['max_iter'],
                        num_batches=args['num_batches'],
                        samples_per_batch=args['samples_per_batch'])
    postprocess_time = time.time() - postprocess_start
    
    # Compute exact energy for JSON output
    fci_start = time.time()
    chemical_potential = args['filling_factor'] * args['onsite']
    hcore, eri = skqd_helpers.siam_hamiltonian(
        args['num_orbs'], args['hopping'], args['onsite'],
        args['hybridization'], chemical_potential
    )
    exact_energy = exact_siam_energy(hcore, eri, args['num_orbs'])
    fci_time = time.time() - fci_start
    print(f"FCI exact energy computed in {fci_time:.2f}s")
    sqd_energy = result[-1]
    error_pct = abs((sqd_energy - exact_energy) / exact_energy) * 100
    
    total_time = time.time() - start_time
    
    # Output JSON result line for parsing by study runner
    result_json = {
        'sqd_energy': sqd_energy,
        'exact_energy': exact_energy,
        'error_pct': error_pct,
        'total_time': total_time,
        'pre_time': pre_time,
        'build_time': build_time,
        'transpile_time': transpile_time,
        'exec_time': exec_time,
        'postprocess_time': postprocess_time,
        'fci_time': fci_time,
        'num_circuits': len(circuits),
        'avg_depth': avg_depth,
    }
    print(f"SKQD_RESULT_JSON:{json.dumps(result_json)}")
    
    print("\n" + "=" * 60)
    print(f"SKQD Test Suite Complete (total: {total_time:.1f}s, postprocess: {postprocess_time:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
