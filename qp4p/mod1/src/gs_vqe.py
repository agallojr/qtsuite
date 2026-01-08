"""
VQE ground state energy calculation using shots-based simulation.
"""

#pylint: disable=protected-access, invalid-name

import argparse
import json
import sys
from qiskit.circuit.library import TwoLocal, efficient_su2
from qp4p_chem import MOLECULES, build_molecular_hamiltonian_fci
from qp4p_vqe import run_vqe_optimization
from qp4p_args import add_noise_args, add_backend_args


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE ground state energy calculation for H2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python gs.py
  python gs.py --shots 4096
  python gs.py --molecule LiH --bond-length 1.6
  python gs.py --molecule HeH+ --shots 2048 --t1 50 --t2 30
""")
    parser.add_argument("--molecule", type=str, default="H2",
                        choices=list(MOLECULES.keys()),
                        help=f"Molecule to simulate (default: H2, available: {list(MOLECULES.keys())})")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Number of shots for expectation value estimation (default: 8192)")
    add_noise_args(parser)
    add_backend_args(parser)
    parser.add_argument("--bond-length", type=float, default=None,
                        help="Bond length in Angstroms (default: molecule-specific)")
    parser.add_argument("--ansatz", type=str, default="EfficientSU2",
                        choices=["TwoLocal", "EfficientSU2"],
                        help="Ansatz type (default: EfficientSU2)")
    parser.add_argument("--entanglement", type=str, default="linear",
                        choices=["linear", "full"],
                        help="Entanglement pattern (default: linear)")
    parser.add_argument("--reps", type=int, default=2,
                        help="Number of ansatz repetitions (default: 2)")
    parser.add_argument("--method", type=str, default="manual",
                        choices=["qiskit", "manual"],
                        help="VQE method: qiskit (statevector) or manual (shots-based SPSA) (default: manual)")
    parser.add_argument("--optimizer", type=str, default="SPSA",
                        choices=["COBYLA", "SLSQP", "SPSA"],
                        help="Optimizer to use (default: SPSA)")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Number of optimization attempts for manual method (default: 5, ignored for qiskit)")
    parser.add_argument("--maxiter", type=int, default=500,
                        help="Max iterations per attempt (default: 500)")
    args = parser.parse_args()

    # Build Hamiltonian
    hamil_qop, fci_energy, scf_energy, mol_info = build_molecular_hamiltonian_fci(args.molecule, args.bond_length)
    
    # Create ansatz based on user choice
    num_qubits = hamil_qop.num_qubits
    if args.ansatz == "TwoLocal":
        ansatz = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cx',
                         entanglement=args.entanglement, reps=args.reps)
    else:  # EfficientSU2
        ansatz = efficient_su2(num_qubits, entanglement=args.entanglement, reps=args.reps)
    
    # Validate method-specific arguments
    if args.method == 'qiskit':
        if args.shots != 8192:
            print("Warning: --shots is ignored for qiskit method (uses statevector)", file=sys.stderr)
        if args.attempts != 5:
            print("Warning: --attempts is ignored for qiskit method", file=sys.stderr)
    elif args.method == 'manual':
        if args.optimizer != 'SPSA':
            print(f"Warning: manual method uses SPSA optimizer, ignoring --optimizer {args.optimizer}", file=sys.stderr)
    
    # Run VQE using unified helper
    vqe_results = run_vqe_optimization(
        hamiltonian=hamil_qop,
        ansatz=ansatz,
        optimizer=args.optimizer,
        method=args.method,
        maxiter=args.maxiter,
        shots=args.shots if args.method == 'manual' else None,
        t1=args.t1,
        t2=args.t2,
        backend=args.backend,
        coupling_map=args.coupling_map,
        num_attempts=args.attempts
    )
    
    # Compute error metrics
    vqe_energy = vqe_results["energy"]
    vqe_error = abs(vqe_energy - fci_energy)
    chemical_accuracy = 0.0015936  # Hartree
    
    # Build results dict
    results = {
        "molecule": mol_info,
        "reference_energies": {
            "fci_hartree": fci_energy,
            "scf_hartree": scf_energy
        },
        "vqe": vqe_results,
        "analysis": {
            "vqe_energy_hartree": vqe_energy,
            "error_hartree": vqe_error,
            "error_vs_chemical_accuracy": vqe_error / chemical_accuracy,
            "within_chemical_accuracy": bool(vqe_error < chemical_accuracy)
        },
        "run_config": {
            "shots": args.shots,
            "t1_us": args.t1,
            "t2_us": args.t2,
            "backend": args.backend
        }
    }
    
    print(json.dumps(results, indent=2))

    

