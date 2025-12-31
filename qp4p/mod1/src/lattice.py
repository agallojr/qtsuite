#!/usr/bin/env python3
"""
Ising Model VQE Solver for Square Lattices

This script builds an Ising model on a 2D square lattice and uses VQE (Variational
Quantum Eigensolver) to find the ground state energy and spin configuration.
"""

#pylint: disable=broad-exception-caught

import re
import argparse
import sys
import json
import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit_algorithms import VQE
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit_nature.second_q.hamiltonians import IsingModel
from qiskit_nature.second_q.hamiltonians.lattices import SquareLattice
from qiskit_nature.second_q.hamiltonians.lattices.boundary_condition import BoundaryCondition
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator

from qp4p_circuit import build_noise_model


def get_ansatz(num_qubits, ansatz_type='standard'):
    """
    Create an ansatz circuit for VQE.
    
    Args:
        num_qubits: Number of qubits
        ansatz_type: Type of ansatz ('simple' or 'standard')
        
    Returns:
        Ansatz circuit
    """
    if ansatz_type == 'simple':
        return TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cz',
                       entanglement='linear', reps=2)
    elif ansatz_type == 'standard':
        return TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cz',
                       entanglement='full', reps=3)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def get_optimizer(optimizer_name, maxiter=100):
    """
    Create an optimizer for VQE.
    
    Args:
        optimizer_name: Name of optimizer ('COBYLA', 'SLSQP', 'SPSA')
        maxiter: Maximum iterations
        
    Returns:
        Optimizer instance
    """
    if optimizer_name == 'COBYLA':
        return COBYLA(maxiter=maxiter)
    if optimizer_name == 'SLSQP':
        return SLSQP(maxiter=maxiter)
    elif optimizer_name == 'SPSA':
        return SPSA(maxiter=maxiter)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def spinop_to_pauli(hamiltonian_op, num_qubits):
    """
    Convert SpinOp to SparsePauliOp.
    
    Args:
        hamiltonian_op: SpinOp Hamiltonian
        num_qubits: Number of qubits
        
    Returns:
        SparsePauliOp
    """
    pauli_list = []
    coeffs = []
    
    for label_str, coeff in hamiltonian_op.items():
        pauli_str = ['I'] * num_qubits
        
        terms = str(label_str).replace('*', ' ').replace('+', ' ').replace('-', ' ').split()
        
        for term in terms:
            if not term or term.isspace():
                continue
            
            match = re.match(r'([XYZ])_(\d+)', term)
            if match:
                op_type = match.group(1)
                idx = int(match.group(2))
                pauli_str[idx] = op_type
        
        pauli_list.append(''.join(reversed(pauli_str)))
        coeffs.append(complex(coeff).real)
    
    return SparsePauliOp(pauli_list, coeffs)




def run_vqe(args):
    """
    Main VQE execution function.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with all results for JSON output
    """
    output = {
        "lattice": {},
        "hamiltonian": {},
        "vqe_config": {},
        "vqe_results": {},
        "ground_state": {}
    }
    
    boundary = BoundaryCondition.PERIODIC \
        if args.boundary == 'periodic' else BoundaryCondition.OPEN
    square_lattice = SquareLattice(rows=args.rows, cols=args.cols, boundary_condition=boundary)
    
    ising_model = IsingModel(
        square_lattice.uniform_parameters(
            uniform_interaction=args.interaction,
            uniform_onsite_potential=args.onsite,
        ),
    )
    
    output["lattice"] = {
        "rows": int(args.rows),
        "cols": int(args.cols),
        "num_nodes": int(square_lattice.num_nodes),
        "boundary": args.boundary,
        "interaction": float(args.interaction),
        "onsite_potential": float(args.onsite)
    }
    
    coupling_matrix = ising_model.coupling_matrix()
    graph_array = ising_model.interaction_matrix()
    
    output["hamiltonian"] = {
        "interaction_matrix_shape": list(graph_array.shape),
        "non_zero_interactions": int(np.count_nonzero(graph_array))
    }
    
    hamiltonian_op = ising_model.second_q_op()
    num_qubits = square_lattice.num_nodes
    
    qubit_hamiltonian = spinop_to_pauli(hamiltonian_op, num_qubits)
    
    output["hamiltonian"]["num_qubits"] = int(qubit_hamiltonian.num_qubits)
    output["hamiltonian"]["num_terms"] = int(len(qubit_hamiltonian))
    
    ansatz = get_ansatz(num_qubits, args.ansatz)
    optimizer = get_optimizer(args.optimizer, args.maxiter)
    
    output["vqe_config"] = {
        "ansatz_type": args.ansatz,
        "ansatz_parameters": int(ansatz.num_parameters),
        "ansatz_depth": int(ansatz.depth()),
        "optimizer": args.optimizer,
        "max_iterations": int(args.maxiter),
        "random_seed": int(args.seed),
        "simulator": "StatevectorEstimator"
    }
    
    # Use StatevectorEstimator for reliable VQE compatibility
    # Note: T1/T2 noise parameters accepted but not applied due to estimator limitations
    if args.t1 > 0 and args.t2 > 0:
        output["vqe_config"]["noise"] = {
            "t1_seconds": args.t1,
            "t2_seconds": args.t2,
            "t1_microseconds": args.t1 * 1e6,
            "t2_microseconds": args.t2 * 1e6,
            "applied": False,
            "note": "StatevectorEstimator does not support noise - parameters ignored"
        }
    
    estimator = StatevectorEstimator()
    
    energy_history = []
    
    def callback(eval_count, params, mean, std):
        energy_history.append(float(mean))
    
    algorithm_globals.random_seed = args.seed
    
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, callback=callback)
    result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
    
    output["vqe_results"] = {
        "ground_state_energy": float(result.eigenvalue),
        "optimizer_evaluations": int(result.cost_function_evals),
        "energy_history": energy_history
    }
    
    optimal_circuit = ansatz.assign_parameters(result.optimal_parameters)
    ground_state = Statevector(optimal_circuit)
    probs = ground_state.probabilities()
    most_likely_state = np.argmax(probs)
    max_prob = probs[most_likely_state]
    
    ground_state_bitstring = format(most_likely_state, f'0{num_qubits}b')
    
    output["ground_state"] = {
        "most_likely_bitstring": ground_state_bitstring,
        "most_likely_probability": float(max_prob),
        "state_is_mixed": bool(max_prob < 0.5)
    }
    
    if max_prob < 0.5:
        ground_state_spins = []
        for i in range(num_qubits):
            pauli_z = ['I'] * num_qubits
            pauli_z[num_qubits - 1 - i] = 'Z'
            z_op = Pauli(''.join(pauli_z))
            expectation = ground_state.expectation_value(z_op).real
            ground_state_spins.append(1 if expectation > 0 else -1)
        
        ground_state_bitstring = ''.join(['0' if s == 1 else '1' for s in ground_state_spins])
        output["ground_state"]["method"] = "expectation_values"
    else:
        ground_state_spins = [1 if bit == '0' else -1 for bit in ground_state_bitstring]
        output["ground_state"]["method"] = "most_likely_state"
    
    output["ground_state"]["spin_configuration"] = [int(s) for s in ground_state_spins]
    
    # Calculate classical ground state energy
    # For ferromagnetic (interaction < 0): all spins aligned
    # For antiferromagnetic (interaction > 0): alternating pattern (if possible)
    num_edges = int(np.count_nonzero(graph_array)) // 2  # Divide by 2 since matrix is symmetric
    
    if args.interaction < 0:
        # Ferromagnetic: all spins aligned gives minimum energy
        classical_energy = args.interaction * num_edges
    else:
        # Antiferromagnetic: best case is alternating (bipartite graph)
        # For square lattice, this is possible; energy = interaction * num_edges
        classical_energy = args.interaction * num_edges
    
    # Calculate fidelity metrics
    vqe_energy = float(result.eigenvalue)
    energy_error = abs(vqe_energy - classical_energy)
    relative_error = abs(energy_error / classical_energy) if classical_energy != 0 else 0.0
    
    # Check if VQE found correct spin pattern
    if args.interaction < 0:
        # For ferromagnetic, all spins should be same
        all_same = all(s == ground_state_spins[0] for s in ground_state_spins)
        pattern_correct = all_same
    else:
        # For antiferromagnetic on square lattice, check if neighbors are opposite
        pattern_correct = None  # More complex to verify
    
    output["classical_ground_state"] = {
        "energy": float(classical_energy),
        "num_edges": int(num_edges),
        "description": "All spins aligned" if args.interaction < 0 else "Alternating pattern (bipartite)"
    }
    
    output["fidelity"] = {
        "energy_error": float(energy_error),
        "relative_error": float(relative_error),
        "relative_error_percent": float(relative_error * 100),
        "vqe_energy": float(vqe_energy),
        "classical_energy": float(classical_energy),
        "pattern_correct": pattern_correct if pattern_correct is None else bool(pattern_correct)
    }
    
    # Add visualization data for postprocessing
    # Convert complex matrices to real (Ising matrices should be real anyway)
    output["visualization_data"] = {
        "coupling_matrix": np.real(coupling_matrix).tolist(),
        "interaction_matrix": np.real(graph_array).tolist(),
        "energy_history": energy_history,
        "ground_state_spins": [int(s) for s in ground_state_spins]
    }
    
    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ising Model VQE Solver for Square Lattices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --rows 3 --cols 3
  %(prog)s --rows 4 --cols 4 --ansatz standard --optimizer SLSQP
  %(prog)s --rows 3 --cols 3 --interaction 1.0 --boundary periodic
  %(prog)s --rows 4 --cols 4 --t1 50e-6 --t2 70e-6
        """
    )
    
    parser.add_argument('--rows', type=int, default=4,
        help='Number of rows in lattice (default: 4)')
    parser.add_argument('--cols', type=int, default=4,
        help='Number of columns in lattice (default: 4)')
    parser.add_argument('--interaction', type=float, default=-1.0,
        help='Uniform interaction strength (default: -1.0, ferromagnetic)')
    parser.add_argument('--onsite', type=float, default=0.0,
        help='Uniform onsite potential (default: 0.0)')
    parser.add_argument('--boundary', choices=['open', 'periodic'], default='open',
        help='Boundary condition (default: open)')
    parser.add_argument('--ansatz', choices=['simple', 'standard'], default='simple',
        help='Ansatz type: simple (linear, reps=2) or standard (full, reps=3) (default: simple)')
    parser.add_argument('--optimizer', choices=['COBYLA', 'SLSQP', 'SPSA'], default='COBYLA',
        help='Optimizer to use (default: COBYLA)')
    parser.add_argument('--maxiter', type=int, default=100,
        help='Maximum optimizer iterations (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument('--t1', type=float, default=0.0,
        help='T1 relaxation time in seconds (default: 0.0, no noise)')
    parser.add_argument('--t2', type=float, default=0.0,
        help='T2 dephasing time in seconds (default: 0.0, no noise)')
    
    args = parser.parse_args()

    
    if args.rows < 2 or args.cols < 2:
        error_output = {"error": "Lattice must be at least 2x2"}
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)
    
    try:
        output = run_vqe(args)
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        error_output = {
            "error": str(e),
            "type": type(e).__name__
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
