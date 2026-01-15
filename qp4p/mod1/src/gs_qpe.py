"""
QPE (Quantum Phase Estimation) ground state energy calculation.

Uses QPE to extract the ground state energy eigenvalue from the
time evolution operator U = e^{-iHt}.
"""

#pylint: disable=protected-access, invalid-name, too-many-locals, too-many-arguments

import argparse
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFTGate

from qp4p_circuit import run_circuit, BASIS_GATES
from qp4p_chem import MOLECULES, build_molecular_hamiltonian_fci
from qp4p_args import add_noise_args, add_backend_args
from qp4p_output import create_standardized_output, output_json


def build_qpe_circuit(hamiltonian_matrix: np.ndarray, num_ancilla: int = 4, 
                      evolution_time: float = 1.0) -> QuantumCircuit:
    """
    Build QPE circuit for ground state energy estimation.
    
    Following Qiskit textbook pattern:
    - Counting qubits: 0 to num_ancilla-1
    - State qubits: num_ancilla onwards
    - Controlled-U^(2^k) applied with qubit k as control
    
    Args:
        hamiltonian_matrix: Hamiltonian as numpy matrix
        num_ancilla: Number of ancilla qubits for phase precision
        evolution_time: Time parameter t in U = e^{-iHt}
    
    Returns:
        QuantumCircuit for QPE
    """
    num_state_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
    total_qubits = num_ancilla + num_state_qubits
    
    # Build unitary U = e^{+iHt} so that eigenvalue is e^{iEt} = e^{2πiφ}
    # This gives φ = Et/(2π), making phase directly proportional to energy
    U_matrix = expm(1j * hamiltonian_matrix * evolution_time)
    
    # Get ground state eigenvector for initial state preparation
    _, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    ground_state = eigenvectors[:, 0]
    
    # Create circuit: counting qubits first, then state qubits
    qc = QuantumCircuit(total_qubits, num_ancilla)
    
    # Initialize state register to ground state (qubits num_ancilla to total_qubits-1)
    state_qubits = list(range(num_ancilla, total_qubits))
    qc.initialize(ground_state, state_qubits)
    
    # Apply H to all counting qubits
    for i in range(num_ancilla):
        qc.h(i)
    
    # Apply controlled-U^(2^k) operations
    # Following textbook: qubit k controls U^(2^k)
    # But Qiskit reverses qubit order, so qubit 0 is LSB
    for k in range(num_ancilla):
        power = 2 ** k
        U_power = np.linalg.matrix_power(U_matrix, power)
        cu_gate = Operator(U_power).to_instruction().control(1)
        qc.append(cu_gate, [k] + state_qubits)
    
    # Apply inverse QFT to counting qubits
    qft_inv = QFTGate(num_ancilla).inverse()
    qc.append(qft_inv, range(num_ancilla))
    
    # Measure counting qubits
    qc.measure(range(num_ancilla), range(num_ancilla))
    
    return qc


def run_qpe(hamiltonian_matrix: np.ndarray, num_ancilla: int = 4, 
            evolution_time: float = None, shots: int = 1024,
            t1: float = None, t2: float = None,
            backend: str = None, coupling_map: str = "default") -> dict:
    """
    Run QPE to estimate ground state energy.
    
    Args:
        hamiltonian_matrix: Hamiltonian as numpy matrix
        num_ancilla: Number of ancilla qubits (determines precision)
        evolution_time: Time parameter for U = e^{-iHt}. If None, auto-computed.
        shots: Number of measurement shots
        t1, t2: Noise parameters
        coupling_map: Coupling map for transpilation
    
    Returns:
        dict with QPE results
    """
    # Get eigenvalues to determine energy shift and evolution time
    eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix)
    E_min = eigenvalues[0]  # Ground state energy
    E_max = eigenvalues[-1]
    
    # Shift Hamiltonian so all eigenvalues are in [0, E_range]
    # This ensures phases are in [0, 1) without wrapping issues
    energy_shift = E_min - 0.1  # Small buffer
    H_shifted = hamiltonian_matrix - energy_shift * np.eye(hamiltonian_matrix.shape[0])
    
    # Shifted eigenvalues
    E_range = E_max - energy_shift
    
    # Set evolution_time so max phase < 1 (i.e., E_max * t / 2π < 1)
    if evolution_time is None:
        evolution_time = 2 * np.pi * 0.9 / E_range  # 0.9 factor for safety margin
    
    # Build QPE circuit with shifted Hamiltonian
    qc = build_qpe_circuit(H_shifted, num_ancilla, evolution_time)
    
    # Transpile to basis gates for Aer simulator
    qc_transpiled = transpile(qc, basis_gates=BASIS_GATES)
    run_result = run_circuit(qc_transpiled, shots=shots, t1=t1, t2=t2, backend=backend, 
                        coupling_map=coupling_map)
    counts = run_result["counts"]
    backend_info = run_result["backend_info"]
    transpiled_stats = run_result["transpiled_stats"]
    
    # Extract phase from measurement results
    # Qiskit's measurement is little-endian: bitstring[0] is the last qubit measured
    # For QPE, we measure qubits 0..n-1, so bitstring is already in correct order
    measured_phases = {}
    for bitstring, count in counts.items():
        phase_int = int(bitstring, 2)
        phase = phase_int / (2 ** num_ancilla)  # Phase in [0, 1)
        measured_phases[phase] = measured_phases.get(phase, 0) + count
    
    # Find most likely phase
    best_phase = max(measured_phases, key=measured_phases.get)
    
    # Convert phase to energy and undo the shift
    # QPE measures φ where e^{-iE't} = e^{-2πiφ}, so E' = 2πφ/t
    # Then E = E' + energy_shift
    shifted_energy = best_phase * 2 * np.pi / evolution_time
    estimated_energy = shifted_energy + energy_shift
    
    result = {
        "estimated_energy": float(estimated_energy),
        "best_phase": float(best_phase),
        "evolution_time": float(evolution_time),
        "energy_shift": float(energy_shift),
        "num_ancilla": num_ancilla,
        "phase_counts": {str(k): v for k, v in sorted(measured_phases.items())},
        "circuit_stats": {
            "num_qubits": qc.num_qubits,
            "depth": qc.depth()
        },
        "transpiled_stats": transpiled_stats,
        "backend_info": backend_info
    }
    
    return result


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QPE ground state energy calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python gs_qpe.py
  python gs_qpe.py --molecule H2 --ancilla 6
  python gs_qpe.py --molecule HeH+ --shots 4096
""")
    parser.add_argument("--molecule", type=str, default="H2",
                        choices=list(MOLECULES.keys()),
                        help="Molecule to simulate (default: H2)")
    parser.add_argument("--ancilla", type=int, default=6,
                        help="Number of ancilla qubits for phase precision (default: 6)")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Number of shots (default: 8192)")
    add_noise_args(parser)
    add_backend_args(parser)
    parser.add_argument("--bond-length", type=float, default=None,
                        help="Bond length in Angstroms (default: molecule-specific)")
    parser.add_argument("--evolution-time", type=float, default=None,
                        help="Evolution time parameter (default: auto-computed)")
    args = parser.parse_args()

    # Build Hamiltonian (as matrix for QPE)
    hamiltonian, fci_energy, scf_energy, mol_info = \
        build_molecular_hamiltonian_fci(args.molecule, args.bond_length, return_matrix=True)
    
    # Run QPE
    qpe_results = run_qpe(
        hamiltonian,
        num_ancilla=args.ancilla,
        evolution_time=args.evolution_time,
        shots=args.shots,
        t1=args.t1,
        t2=args.t2,
        backend=args.backend,
        coupling_map=args.coupling_map
    )
    
    # Compute error metrics
    qpe_energy = qpe_results["estimated_energy"]
    qpe_error = abs(qpe_energy - fci_energy)
    chemical_accuracy = 0.0015936  # Hartree
    
    output = create_standardized_output(
        algorithm="qpe",
        script_name="gs_qpe.py",
        problem={
            "molecule": {
                "name": args.molecule,
                "basis": mol_info["basis"],
                "bond_length": mol_info["bond_length"],
                "num_qubits": mol_info["num_qubits"]
            },
            "reference_energies": {
                "scf_hartree": scf_energy,
                "fci_hartree": fci_energy
            }
        },
        config={
            "num_ancilla": args.ancilla,
            "evolution_time": qpe_results["evolution_time"],
            "shots": args.shots,
            "t1_us": args.t1,
            "t2_us": args.t2,
            "backend": args.backend
        },
        results=qpe_results,
        metrics={
            "qpe_energy_hartree": qpe_energy,
            "error_hartree": qpe_error,
            "error_vs_chemical_accuracy": qpe_error / chemical_accuracy,
            "within_chemical_accuracy": bool(qpe_error < chemical_accuracy)
        }
    )
    
    output_json(output)
