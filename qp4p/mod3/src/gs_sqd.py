"""
Sample-based Quantum Diagonalization (SQD) for molecular systems.

Based on "Chemistry Beyond the Scale of Exact Diagonalization on a 
Quantum-Centric Supercomputer" (arXiv:2405.05068)

Uses qiskit-addon-sqd to perform iterative configuration recovery and 
subspace diagonalization for ground state energy estimation.
"""

#pylint: disable=invalid-name, protected-access, too-many-locals

import argparse
import json
import numpy as np

from pyscf import gto, scf, fci, ao2mo
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_addon_sqd.counts import generate_counts_uniform, counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_fermion

from qp4p_args import add_noise_args, add_backend_args
from qp4p_circuit import build_noise_model
from qp4p_chem import MOLECULES


def get_counts_from_hf_circuit(num_orbitals: int, num_elec_a: int, num_elec_b: int,
                                seed: int, shots: int, backend_name: str = None,
                                t1: float = None, t2: float = None):
    """
    Get counts from Hartree-Fock state with small perturbations.
    
    Args:
        num_orbitals: Number of spatial orbitals
        num_elec_a: Number of alpha electrons
        num_elec_b: Number of beta electrons
        seed: Random seed for perturbations
        shots: Number of measurement shots
        backend_name: Fake backend name (optional)
        t1: T1 relaxation time in µs (optional)
        t2: T2 dephasing time in µs (optional)
    
    Returns:
        counts: Measurement counts dictionary
    """
    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circuit = QuantumCircuit(qubits)

    # Prepare Hartree-Fock state
    total_electrons = num_elec_a + num_elec_b
    for i in range(total_electrons):
        circuit.x(i)

    # Add small random rotations to break symmetry
    np.random.seed(seed)
    for i in range(2 * num_orbitals):
        if np.random.random() < 0.3:
            circuit.ry(np.random.uniform(0, np.pi/8), i)

    circuit.measure_all()

    # Set up backend with noise if specified
    if backend_name or (t1 is not None and t2 is not None):
        noise_model, fake_backend, _ = build_noise_model(t1, t2, backend_name, "default")
        
        if fake_backend:
            transpiled_circuit = transpile(circuit, backend=fake_backend, optimization_level=1)
            backend = AerSimulator.from_backend(fake_backend)
        else:
            transpiled_circuit = transpile(circuit, optimization_level=1)
            backend = AerSimulator()
        
        if noise_model:
            result = backend.run(transpiled_circuit, shots=shots, noise_model=noise_model).result()
        else:
            result = backend.run(transpiled_circuit, shots=shots).result()
    else:
        backend = AerSimulator(method='extended_stabilizer')
        transpiled_circuit = transpile(circuit, backend=backend, optimization_level=1)
        result = backend.run(transpiled_circuit, shots=shots).result()

    return result.get_counts()


def run_sqd_pipeline(
    num_orbitals: int,
    core_hamiltonian: np.ndarray,
    electron_repulsion_integrals: np.ndarray,
    nuclear_repulsion_energy: float,
    num_alpha: int,
    num_beta: int,
    open_shell: bool,
    spin_sq: int,
    iterations: int,
    num_batches: int,
    samples_per_batch: int,
    rng_seed: int,
    shots: int,
    synthetic_counts: bool,
    backend_name: str = None,
    t1: float = None,
    t2: float = None,
):
    """
    Run the SQD pipeline.
    
    Returns:
        energy_hist: Energy history per iteration and batch (Hartree)
        spin_sq_hist: Spin expectation per iteration and batch
        final_energy: Final converged energy
        final_error: Error vs exact (if available)
    """
    rng = np.random.default_rng(rng_seed)
    
    if synthetic_counts:
        counts = generate_counts_uniform(shots, num_orbitals * 2, rand_seed=rng)
    else:
        counts = get_counts_from_hf_circuit(
            num_orbitals, num_alpha, num_beta, rng_seed, shots,
            backend_name, t1, t2
        )

    # Convert counts into bitstring and probability arrays
    bitstring_matrix_full, probs_array_full = counts_to_arrays(counts)

    # Initialize histories
    energy_hist = np.zeros((iterations, num_batches))
    spin_sq_hist = np.zeros((iterations, num_batches))
    avg_occupancy = None

    for i in range(iterations):
        # Configuration recovery
        if avg_occupancy is None:
            bitstring_matrix_tmp = bitstring_matrix_full
            probs_array_tmp = probs_array_full
        else:
            bitstring_matrix_tmp, probs_array_tmp = recover_configurations(
                bitstring_matrix_full,
                probs_array_full,
                avg_occupancy,
                num_alpha,
                num_beta,
                rand_seed=rng,
            )

        # Post-select and subsample
        bitstring_matrix_ps, probs_array_ps = postselect_by_hamming_right_and_left(
            bitstring_matrix_tmp,
            probs_array_tmp,
            hamming_right=num_alpha,
            hamming_left=num_beta,
        )
        batches = subsample(
            bitstring_matrix_ps,
            probs_array_ps,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            rand_seed=rng,
        )

        # Run eigenstate solvers
        e_tmp = np.zeros(num_batches)
        s_tmp = np.zeros(num_batches)
        occs_tmp = []
        
        for j in range(num_batches):
            strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
            subspace_dim = len(strs_a) * len(strs_b)
            
            energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
                batches[j],
                core_hamiltonian,
                electron_repulsion_integrals,
                open_shell=open_shell,
                spin_sq=spin_sq,
                max_cycle=200
            )
            energy_sci += nuclear_repulsion_energy
            e_tmp[j] = energy_sci
            s_tmp[j] = spin
            occs_tmp.append(avg_occs)

        # Combine batch results
        avg_occupancy = tuple(np.mean(occs_tmp, axis=0))
        energy_hist[i, :] = e_tmp
        spin_sq_hist[i, :] = s_tmp

    final_energy = float(np.min(energy_hist[-1, :]))
    
    return energy_hist, spin_sq_hist, final_energy


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample-based Quantum Diagonalization (SQD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python gs_sqd.py --molecule H2
  python gs_sqd.py --molecule LiH --bond-length 1.6 --iterations 10
  python gs_sqd.py --molecule H2O --backend manila
""")
    parser.add_argument("--molecule", type=str, default="H2",
                        choices=list(MOLECULES.keys()),
                        help=f"Molecule to simulate (default: H2, available: {list(MOLECULES.keys())})")
    parser.add_argument("--bond-length", type=float, default=None,
                        help="Bond length in Angstroms (default: molecule-specific)")
    parser.add_argument("--exact-energy", type=float, default=None,
                        help="Exact reference energy for error calculation (Hartree)")
    parser.add_argument("--spin-sq", type=int, default=0,
                        help="Target S(S+1) value; 0 for singlet (default: 0)")
    parser.add_argument("--open-shell", action="store_true",
                        help="Open-shell system (default: closed-shell)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Configuration-recovery iterations (default: 5)")
    parser.add_argument("--batches", type=int, default=1,
                        help="Number of subspace batches per iteration (default: 1)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Configurations per batch after subsampling (default: 500)")
    parser.add_argument("--shots", type=int, default=10000,
                        help="Measurement shots for circuit (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic uniform counts instead of circuit")
    add_noise_args(parser)
    add_backend_args(parser)
    
    args = parser.parse_args()

    # Build molecule using PySCF
    mol_def = MOLECULES[args.molecule]
    d = args.bond_length if args.bond_length is not None else mol_def.get("default_bond")
    
    mol = gto.Mole()
    mol.atom = mol_def["atoms"].format(d=d) if d else mol_def["atoms"]
    mol.basis = mol_def["basis"]
    mol.charge = mol_def.get("charge", 0)
    mol.spin = mol_def.get("spin", 0)
    mol.build()

    # Run Hartree-Fock
    mf = scf.RHF(mol).run(verbose=0)
    
    # Get exact FCI energy for reference
    fci_solver = fci.FCI(mf)
    exact_fci_energy = fci_solver.kernel()[0]
    
    # Extract molecular integrals in MO basis
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    nuclear_repulsion_energy = mf.energy_nuc()
    num_orbitals = mf.mo_coeff.shape[1]
    num_alpha, num_beta = mol.nelec
    
    # Use FCI energy as exact reference if not provided
    if args.exact_energy is None:
        args.exact_energy = exact_fci_energy
    
    core_hamiltonian = h1e
    electron_repulsion_integrals = h2e

    # Run SQD pipeline
    energy_hist, spin_sq_hist, final_energy = run_sqd_pipeline(
        num_orbitals=num_orbitals,
        core_hamiltonian=core_hamiltonian,
        electron_repulsion_integrals=electron_repulsion_integrals,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        num_alpha=num_alpha,
        num_beta=num_beta,
        open_shell=args.open_shell,
        spin_sq=args.spin_sq,
        iterations=args.iterations,
        num_batches=args.batches,
        samples_per_batch=args.samples,
        rng_seed=args.seed,
        shots=args.shots,
        synthetic_counts=args.synthetic,
        backend_name=args.backend,
        t1=args.t1,
        t2=args.t2,
    )

    # Compute error metrics
    chemical_accuracy = 0.001  # 1 milli-Hartree
    error_hartree = None
    within_chemical_accuracy = None
    
    if args.exact_energy is not None:
        error_hartree = abs(final_energy - args.exact_energy)
        within_chemical_accuracy = bool(error_hartree < chemical_accuracy)

    # Build results dict
    results = {
        "molecule": {
            "name": args.molecule,
            "bond_length": d,
            "basis": mol_def["basis"],
            "num_orbitals": num_orbitals,
            "num_alpha": num_alpha,
            "num_beta": num_beta,
            "nuclear_repulsion_hartree": float(nuclear_repulsion_energy),
            "open_shell": args.open_shell,
            "spin_sq": args.spin_sq
        },
        "sqd_config": {
            "iterations": args.iterations,
            "num_batches": args.batches,
            "samples_per_batch": args.samples,
            "shots": args.shots,
            "synthetic_counts": args.synthetic,
            "rng_seed": args.seed
        },
        "convergence": {
            "energy_per_iteration": [float(np.min(energy_hist[i, :])) for i in range(args.iterations)],
            "energy_std_per_iteration": [float(np.std(energy_hist[i, :])) for i in range(args.iterations)],
            "spin_sq_per_iteration": [float(np.mean(spin_sq_hist[i, :])) for i in range(args.iterations)]
        },
        "results": {
            "sqd_energy_hartree": final_energy,
            "exact_energy_hartree": args.exact_energy,
            "error_hartree": error_hartree,
            "within_chemical_accuracy": within_chemical_accuracy
        },
        "run_config": {
            "t1_us": args.t1,
            "t2_us": args.t2,
            "backend": args.backend,
            "coupling_map": args.coupling_map
        }
    }

    print(json.dumps(results, indent=2))
