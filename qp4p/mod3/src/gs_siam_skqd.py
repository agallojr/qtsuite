"""
Sample-based Krylov Quantum Diagonalization (SKQD) for SIAM.

Implements SKQD for the Single Impurity Anderson Model (SIAM) - a fundamental
model in condensed matter physics for magnetic impurities coupled to conduction
electrons. Uses Krylov subspace methods to enhance SQD sampling.

Based on quantum-classical hybrid algorithm combining:
- Krylov basis generation via Trotterized time evolution
- Bitstring sampling from quantum circuits
- Classical subspace diagonalization (SQD)
"""

#pylint: disable=invalid-name, protected-access, too-many-locals

import argparse
import json
import time
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
from qiskit_addon_sqd.fermion import solve_fermion

from qp4p_args import add_noise_args, add_backend_args
from qp4p_chem import build_siam_hamiltonian
from qp4p_circuit import build_noise_model


def construct_krylov_circuits(num_orbs: int, h1e, h2e, dt: float, krylov_dim: int):
    """Build Krylov basis circuits using simple first-order Trotterization."""
    num_qubits = 2 * num_orbs
    impurity_index = (num_orbs - 1) // 2
    
    circuits = []
    for k in range(krylov_dim):
        qc = QuantumCircuit(num_qubits)
        
        # Prepare initial state (half-filled)
        occ_num = num_orbs // 2
        for i in range(occ_num):
            qc.x(i)
            qc.x(num_orbs + i)
        
        # Apply time evolution k times using improved Trotter
        if k > 0:
            num_trotter_steps = k
            for _ in range(num_trotter_steps):
                # Two-body evolution for half time (impurity only)
                if abs(h2e[impurity_index, impurity_index, impurity_index, impurity_index]) > 1e-10:
                    angle = -0.5 * dt * h2e[impurity_index, impurity_index, impurity_index, impurity_index]
                    qc.cp(angle, impurity_index, num_orbs + impurity_index)
                
                # One-body evolution for full time
                # Apply as single-qubit rotations (diagonal terms) and two-qubit gates (off-diagonal)
                for i in range(num_orbs):
                    # Diagonal terms
                    if abs(h1e[i, i]) > 1e-10:
                        angle = -dt * h1e[i, i]
                        qc.rz(angle, i)
                        qc.rz(angle, num_orbs + i)
                    
                    # Off-diagonal hopping terms (simplified - just nearest neighbor)
                    for j in range(i+1, num_orbs):
                        if abs(h1e[i, j]) > 1e-10:
                            # Use RXX gate for hopping
                            angle = -dt * h1e[i, j]
                            qc.rxx(angle, i, j)
                            qc.rxx(angle, num_orbs + i, num_orbs + j)
                
                # Two-body evolution for half time (impurity only)
                if abs(h2e[impurity_index, impurity_index, impurity_index, impurity_index]) > 1e-10:
                    angle = -0.5 * dt * h2e[impurity_index, impurity_index, impurity_index, impurity_index]
                    qc.cp(angle, impurity_index, num_orbs + impurity_index)
        
        qc.measure_all()
        circuits.append(qc)
    
    return circuits


def execute_circuits_with_noise(circuits, shots: int, noise: float, backend_name: str = None,
                                 t1: float = None, t2: float = None, opt_level: int = 1):
    """
    Execute circuits with optional noise.
    
    Returns:
        counts: Dictionary of bitstring counts
        avg_depth: Average circuit depth after transpilation
    """
    # Set up backend with noise
    if noise > 0:
        noise_model = NoiseModel()
        error_2q = depolarizing_error(noise, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'cp'])
        backend = AerSimulator(noise_model=noise_model)
    elif backend_name or (t1 is not None and t2 is not None):
        noise_model, fake_backend, _ = build_noise_model(t1, t2, backend_name, "default")
        if fake_backend:
            backend = AerSimulator.from_backend(fake_backend)
        else:
            backend = AerSimulator()
    else:
        backend = AerSimulator(method='automatic')
        noise_model = None
    
    # Transpile
    transpiled = transpile(circuits, backend=backend, optimization_level=opt_level)
    
    # Get circuit stats
    depths = [c.depth() for c in transpiled]
    avg_depth = sum(depths) / len(depths) if depths else 0
    
    # Execute all circuits and combine counts
    all_counts = {}
    for circ in transpiled:
        if noise_model:
            result = backend.run(circ, shots=shots, noise_model=noise_model).result()
        else:
            result = backend.run(circ, shots=shots).result()
        counts = result.get_counts()
        for bitstring, count in counts.items():
            all_counts[bitstring] = all_counts.get(bitstring, 0) + count
    
    return all_counts, avg_depth, depths


def run_sqd_postprocess(counts, num_orbs: int, h1e, h2e, max_iter: int, 
                        num_batches: int, samples_per_batch: int):
    """Run SQD post-processing to extract ground state energy."""
    # Convert counts to arrays
    bitstring_matrix_full, probs_array_full = counts_to_arrays(counts)
    
    # Initialize
    energy_history = []
    avg_occupancy = None
    rng = np.random.default_rng(42)
    num_alpha = num_orbs // 2
    num_beta = num_orbs // 2
    
    for _ in range(max_iter):
        # Configuration recovery
        if avg_occupancy is None:
            bitstring_matrix_tmp = bitstring_matrix_full
            probs_array_tmp = probs_array_full
        else:
            bitstring_matrix_tmp, probs_array_tmp = recover_configurations(
                bitstring_matrix_full, probs_array_full,
                avg_occupancy, num_alpha, num_beta, rand_seed=rng
            )
        
        # Post-select and subsample
        bitstring_matrix_ps, probs_array_ps = postselect_by_hamming_right_and_left(
            bitstring_matrix_tmp, probs_array_tmp,
            hamming_right=num_alpha, hamming_left=num_beta
        )
        batches = subsample(
            bitstring_matrix_ps, probs_array_ps,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches, rand_seed=rng
        )
        
        # Solve each batch
        batch_energies = []
        occs_tmp = []
        for batch in batches:
            energy_sci, _, avg_occs, _ = solve_fermion(
                batch, h1e, h2e,
                open_shell=False, spin_sq=0, max_cycle=200
            )
            batch_energies.append(energy_sci)
            occs_tmp.append(avg_occs)
        
        # Update occupancy and record energy
        avg_occupancy = tuple(np.mean(occs_tmp, axis=0))
        min_energy = np.min(batch_energies)
        energy_history.append(min_energy)
    
    return energy_history


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample-based Krylov Quantum Diagonalization (SKQD) for SIAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python gs_siam_skqd.py --num-orbs 8 --krylov-dim 5
  python gs_siam_skqd.py --num-orbs 10 --shots 2048 --noise 0.01
  python gs_siam_skqd.py --num-orbs 8 --backend manila --t1 50 --t2 30
""")
    parser.add_argument("--num-orbs", type=int, default=8,
                        help="Number of spatial orbitals (qubits = 2 × num_orbs, default: 8)")
    parser.add_argument("--krylov-dim", type=int, default=5,
                        help="Krylov dimension (number of basis states, default: 5)")
    parser.add_argument("--dt-mult", type=float, default=1.0,
                        help="Time step multiplier for Trotter evolution (default: 1.0)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Measurement shots per circuit (default: 1024)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Depolarizing noise rate on 2-qubit gates (0-1, default: 0)")
    parser.add_argument("--hopping", type=float, default=1.0,
                        help="Hopping parameter t (default: 1.0)")
    parser.add_argument("--onsite", type=float, default=5.0,
                        help="Onsite Coulomb interaction U (default: 5.0)")
    parser.add_argument("--hybridization", type=float, default=1.0,
                        help="Impurity-bath hybridization V (default: 1.0)")
    parser.add_argument("--filling-factor", type=float, default=-0.5,
                        help="Chemical potential = filling_factor × U (default: -0.5)")
    parser.add_argument("--opt-level", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Transpiler optimization level (default: 1)")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="SQD maximum iterations (default: 10)")
    parser.add_argument("--num-batches", type=int, default=5,
                        help="SQD number of batches (default: 5)")
    parser.add_argument("--samples-per-batch", type=int, default=200,
                        help="SQD samples per batch (default: 200)")
    add_noise_args(parser)
    add_backend_args(parser)
    
    args = parser.parse_args()
    
    start_time = time.time()
    num_qubits = 2 * args.num_orbs
    chemical_potential = args.filling_factor * args.onsite
    
    # Step 1: Build SIAM Hamiltonian
    h1e, h2e, exact_energy, siam_info = build_siam_hamiltonian(
        args.num_orbs, args.hopping, args.onsite,
        args.hybridization, chemical_potential, momentum_basis=True
    )
    
    # Step 2: Build Krylov circuits
    dt = args.dt_mult * np.pi / np.linalg.norm(h1e, ord=2)
    circuits = construct_krylov_circuits(args.num_orbs, h1e, h2e, dt, args.krylov_dim)
    
    # Step 3 & 4: Transpile and execute
    counts, avg_depth, depths = execute_circuits_with_noise(
        circuits, args.shots, args.noise, args.backend, args.t1, args.t2, args.opt_level
    )
    
    # Step 5: SQD post-processing
    energy_history = run_sqd_postprocess(
        counts, args.num_orbs, h1e, h2e,
        args.max_iter, args.num_batches, args.samples_per_batch
    )
    
    # Final results
    skqd_energy = float(energy_history[-1])
    error_abs = abs(skqd_energy - exact_energy)
    error_pct = (error_abs / abs(exact_energy)) * 100 if exact_energy != 0 else 0
    total_time = time.time() - start_time
    
    # Build results dict
    results = {
        "siam_model": siam_info,
        "skqd_config": {
            "krylov_dim": args.krylov_dim,
            "dt_mult": args.dt_mult,
            "dt": float(dt),
            "max_iter": args.max_iter,
            "num_batches": args.num_batches,
            "samples_per_batch": args.samples_per_batch
        },
        "circuit_info": {
            "num_circuits": len(circuits),
            "avg_depth": float(avg_depth),
            "min_depth": int(min(depths)),
            "max_depth": int(max(depths))
        },
        "convergence": {
            "energy_per_iteration": [float(e) for e in energy_history],
            "num_iterations": len(energy_history)
        },
        "results": {
            "skqd_energy": skqd_energy,
            "exact_energy": float(exact_energy),
            "error_abs": error_abs,
            "error_pct": error_pct
        },
        "run_config": {
            "shots": args.shots,
            "noise": args.noise,
            "t1_us": args.t1,
            "t2_us": args.t2,
            "backend": args.backend,
            "coupling_map": args.coupling_map,
            "opt_level": args.opt_level
        },
        "timing": {
            "total_seconds": total_time
        }
    }
    
    print(json.dumps(results, indent=2))
