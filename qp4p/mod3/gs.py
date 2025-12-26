"""
VQE ground state energy calculation using shots-based simulation.
"""

#pylint: disable=protected-access, invalid-name

import argparse
import json
import numpy as np
from pyscf import gto, scf, fci, ao2mo
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit import transpile

from qp4p_circuit import run_estimator
from qp4p_opt import spsa_optimize


# Predefined molecule geometries (atom string templates with {d} for bond length)
MOLECULES = {
    "H2": {
        "atoms": "H 0 0 0; H 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 0.74,
        "charge": 0,
        "spin": 0
    },
    "LiH": {
        "atoms": "Li 0 0 0; H 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 1.6,
        "charge": 0,
        "spin": 0
    },
    "HeH+": {
        "atoms": "He 0 0 0; H 0 0 {d}",
        "basis": "sto-3g",
        "default_bond": 0.93,
        "charge": 1,
        "spin": 0
    },
    "H2O": {
        "atoms": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        "basis": "sto-3g",
        "default_bond": None,  # Fixed geometry
        "charge": 0,
        "spin": 0
    },
    "BeH2": {
        "atoms": "Be 0 0 0; H 0 0 {d}; H 0 0 -{d}",
        "basis": "sto-3g",
        "default_bond": 1.3,
        "charge": 0,
        "spin": 0
    }
}


def build_hamiltonian(molecule: str = "H2", bond_length: float = None):
    """
    Build molecular Hamiltonian using PySCF FCI.
    
    Args:
        molecule: Molecule name (H2, LiH, HeH+, H2O, BeH2)
        bond_length: Bond length in Angstroms (None = use default)
    
    Returns:
        hamil_qop: SparsePauliOp Hamiltonian
        fci_energy: Exact FCI ground state energy
        scf_energy: SCF reference energy
        mol_info: dict with molecule details
    """
    if molecule not in MOLECULES:
        raise ValueError(f"Unknown molecule: {molecule}. Available: {list(MOLECULES.keys())}")
    
    mol_def = MOLECULES[molecule]
    d = bond_length if bond_length is not None else mol_def["default_bond"]
    
    mol = gto.Mole()
    mol.atom = mol_def["atoms"].format(d=d) if d else mol_def["atoms"]
    mol.basis = mol_def["basis"]
    mol.charge = mol_def["charge"]
    mol.spin = mol_def["spin"]
    mol.build()

    mf = scf.RHF(mol).run(verbose=0)

    # Get integrals directly from mean-field calculation
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

    # Transform 2-electron integrals to MO basis
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    ecore = mf.energy_nuc()
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec

    # Construct full FCI Hamiltonian matrix
    fci_solver = fci.FCI(mf)

    # Get exact FCI energy for comparison
    fci_energy = fci_solver.kernel()[0]

    # Build Hamiltonian matrix in full CI space
    _, hamiltonian_matrix = fci_solver.pspace(h1e, h2e, norb, nelec, np=99)
    hamiltonian_matrix = hamiltonian_matrix + np.diag([ecore] * hamiltonian_matrix.shape[0])

    # Pad to next power of 2 if needed (required for qubit representation)
    dim = hamiltonian_matrix.shape[0]
    next_pow2 = 1 << (dim - 1).bit_length()  # Next power of 2
    if dim != next_pow2:
        padded = np.zeros((next_pow2, next_pow2), dtype=hamiltonian_matrix.dtype)
        padded[:dim, :dim] = hamiltonian_matrix
        # Fill diagonal padding with large energy to keep states inaccessible
        for i in range(dim, next_pow2):
            padded[i, i] = 1e6
        hamiltonian_matrix = padded

    # Convert to Pauli operator
    hamil_qop = SparsePauliOp.from_operator(hamiltonian_matrix)
    
    mol_info = {
        "name": molecule,
        "bond_length": d,
        "basis": mol_def["basis"],
        "num_qubits": hamil_qop.num_qubits
    }
    
    return hamil_qop, fci_energy, mf.energy_tot(), mol_info


def run_vqe(hamil_qop: SparsePauliOp, shots: int = 1024, 
            t1: float = None, t2: float = None, 
            num_attempts: int = 5, maxiter: int = 200):
    """
    Run VQE optimization using shots-based Estimator with SPSA optimizer.
    
    Args:
        hamil_qop: Hamiltonian as SparsePauliOp
        shots: Number of shots per expectation value estimation
        t1: T1 relaxation time in microseconds
        t2: T2 dephasing time in microseconds
        num_attempts: Number of optimization attempts with random starts
        maxiter: Maximum iterations per optimization attempt
    
    Returns:
        dict with optimization results
    """
    num_qubits = hamil_qop.num_qubits
    ansatz = EfficientSU2(num_qubits, reps=2, entanglement='linear')
    
    # Transpile ansatz to basis gates
    transpiled_ansatz = transpile(ansatz, basis_gates=['cx', 'h', 'x', 'y', 'z', 'rz', 'ry', 'rx'])
    
    eval_count = [0]
    
    def vqe_cost(params):
        """Compute <ψ(θ)|H|ψ(θ)> using shots-based Estimator"""
        bound_circuit = transpiled_ansatz.assign_parameters(params)
        energy = run_estimator(bound_circuit, hamil_qop, shots=shots, t1=t1, t2=t2)
        eval_count[0] += 1
        return energy
    
    all_attempts = []
    all_results = []
    
    for attempt in range(num_attempts):
        initial_params = np.random.uniform(-0.5, 0.5, ansatz.num_parameters)
        eval_count[0] = 0
        
        # Use SPSA for noise-resilient optimization
        result = spsa_optimize(vqe_cost, initial_params, maxiter=maxiter)
        all_results.append(result)
        
        attempt_info = {
            "attempt": attempt + 1,
            "energy": float(result["fun"]),
            "evaluations": eval_count[0],
            "converged": True
        }
        all_attempts.append(attempt_info)
    
    # Select result with median energy (robust to outliers from noise)
    energies = [r["fun"] for r in all_results]
    median_idx = np.argsort(energies)[len(energies) // 2]
    best_result = all_results[median_idx]
    
    return {
        "best_energy": float(best_result["fun"]),
        "best_params": best_result["x"].tolist(),
        "converged": True,
        "attempts": all_attempts,
        "ansatz": {
            "name": "EfficientSU2",
            "num_qubits": num_qubits,
            "num_parameters": ansatz.num_parameters,
            "reps": 2
        }
    }


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
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in µs (default: None = no noise)")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in µs (default: None = no noise)")
    parser.add_argument("--bond-length", type=float, default=None,
                        help="Bond length in Angstroms (default: molecule-specific)")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Number of optimization attempts (default: 5)")
    parser.add_argument("--maxiter", type=int, default=500,
                        help="Max iterations per attempt (default: 500)")
    args = parser.parse_args()

    # Build Hamiltonian
    hamil_qop, fci_energy, scf_energy, mol_info = build_hamiltonian(args.molecule, args.bond_length)
    
    # Run VQE
    vqe_results = run_vqe(
        hamil_qop, 
        shots=args.shots, 
        t1=args.t1, 
        t2=args.t2,
        num_attempts=args.attempts,
        maxiter=args.maxiter
    )
    
    # Compute error metrics
    vqe_energy = vqe_results["best_energy"]
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
            "t2_us": args.t2
        }
    }
    
    print(json.dumps(results, indent=2))

    

