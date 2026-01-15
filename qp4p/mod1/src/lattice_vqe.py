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
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_nature.second_q.hamiltonians import IsingModel
from qiskit_nature.second_q.hamiltonians.lattices import SquareLattice
from qiskit_nature.second_q.hamiltonians.lattices.boundary_condition import BoundaryCondition

from qp4p_vqe import run_vqe_optimization
from qp4p_args import add_noise_args, add_backend_args
from qp4p_output import create_standardized_output, output_json, output_error


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
    
    # Create ansatz based on user choice
    if args.ansatz == 'TwoLocal':
        ansatz = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cx',
                         entanglement=args.entanglement, reps=args.reps)
    else:  # EfficientSU2
        ansatz = EfficientSU2(num_qubits, entanglement=args.entanglement, reps=args.reps)
    
    # Validate method-specific arguments
    if args.method == 'qiskit':
        if args.shots is not None:
            print("Warning: --shots is ignored for qiskit method (uses statevector)", file=sys.stderr)
        if args.attempts != 5:
            print("Warning: --attempts is ignored for qiskit method", file=sys.stderr)
    elif args.method == 'manual':
        if args.shots is None:
            args.shots = 1024  # Default for manual method
        if args.optimizer != 'SPSA':
            print(f"Warning: manual method uses SPSA optimizer, ignoring --optimizer {args.optimizer}", file=sys.stderr)
    
    # Run VQE using unified helper
    vqe_result = run_vqe_optimization(
        hamiltonian=qubit_hamiltonian,
        ansatz=ansatz,
        optimizer=args.optimizer,
        method=args.method,
        maxiter=args.maxiter,
        shots=args.shots,
        t1=args.t1,
        t2=args.t2,
        backend=args.backend,
        coupling_map=args.coupling_map,
        seed=args.seed,
        num_attempts=args.attempts
    )
    
    output["vqe_config"] = {
        "ansatz_type": args.ansatz,
        "ansatz_parameters": int(vqe_result["ansatz_info"]["num_parameters"]),
        "ansatz_depth": int(vqe_result["ansatz_info"]["depth"]),
        "optimizer": args.optimizer,
        "max_iterations": int(args.maxiter),
        "random_seed": int(args.seed),
        "method": vqe_result["method"]
    }
    
    # Add noise info if applicable
    if args.backend:
        output["vqe_config"]["noise"] = {
            "backend": args.backend,
            "source": "IBM fake backend",
            "applied": True
        }
    elif args.t1 is not None and args.t2 is not None:
        output["vqe_config"]["noise"] = {
            "t1_microseconds": args.t1,
            "t2_microseconds": args.t2,
            "applied": True
        }
    
    output["vqe_results"] = {
        "ground_state_energy": vqe_result["energy"],
        "optimizer_evaluations": vqe_result["evaluations"],
        "energy_history": vqe_result["energy_history"]
    }
    
    # Extract ground state from VQE result
    if vqe_result["optimal_params"]:
        # Bind optimal parameters to ansatz
        params = vqe_result["optimal_params"]
        bound_circuit = ansatz.assign_parameters(params)
        ground_state_vector = Statevector(bound_circuit)
        probs = ground_state_vector.probabilities()
    else:
        # No optimal params, use uniform distribution
        probs = np.ones(2**num_qubits) / (2**num_qubits)
        ground_state_vector = None
    most_likely_state = np.argmax(probs)
    max_prob = probs[most_likely_state]
    
    ground_state_bitstring = format(most_likely_state, f'0{num_qubits}b')
    
    output["ground_state"] = {
        "most_likely_bitstring": ground_state_bitstring,
        "most_likely_probability": float(max_prob),
        "state_is_mixed": bool(max_prob < 0.5)
    }
    
    if max_prob < 0.5 and ground_state_vector is not None:
        ground_state_spins = []
        for i in range(num_qubits):
            pauli_z = ['I'] * num_qubits
            pauli_z[num_qubits - 1 - i] = 'Z'
            z_op = Pauli(''.join(pauli_z))
            expectation = ground_state_vector.expectation_value(z_op).real
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
    vqe_energy = vqe_result["energy"]
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
    visualization_data = {
        "coupling_matrix": np.real(coupling_matrix).tolist(),
        "interaction_matrix": np.real(graph_array).tolist(),
        "energy_history": vqe_result["energy_history"] if vqe_result["energy_history"] else [],
        "ground_state_spins": [int(s) for s in ground_state_spins]
    }
    
    # Return standardized output
    return create_standardized_output(
        algorithm="lattice_vqe",
        script_name="lattice_vqe.py",
        problem={
            "lattice": output["lattice"],
            "hamiltonian": output["hamiltonian"]
        },
        config=output["vqe_config"],
        results={
            "ground_state": output["ground_state"],
            "vqe_results": output["vqe_results"]
        },
        metrics=output["fidelity"],
        visualization_data=visualization_data
    )


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
    parser.add_argument('--ansatz', type=str, default='TwoLocal',
        choices=['TwoLocal', 'EfficientSU2'],
        help='Ansatz type (default: TwoLocal)')
    parser.add_argument('--entanglement', type=str, default='linear',
        choices=['linear', 'full'],
        help='Entanglement pattern (default: linear)')
    parser.add_argument('--reps', type=int, default=2,
        help='Number of ansatz repetitions (default: 2)')
    parser.add_argument('--method', type=str, default='qiskit',
        choices=['qiskit', 'manual'],
        help='VQE method: qiskit (statevector) or manual (shots-based SPSA) (default: qiskit)')
    parser.add_argument('--optimizer', choices=['COBYLA', 'SLSQP', 'SPSA'], default='COBYLA',
        help='Optimizer to use (default: COBYLA)')
    parser.add_argument('--maxiter', type=int, default=100,
        help='Maximum optimizer iterations (default: 100)')
    parser.add_argument('--shots', type=int, default=None,
        help='Number of shots for manual method (default: None = 1024 for manual, ignored for qiskit)')
    parser.add_argument('--attempts', type=int, default=5,
        help='Number of optimization attempts for manual method (default: 5, ignored for qiskit)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed (default: 42)')
    add_noise_args(parser)
    add_backend_args(parser)
    
    args = parser.parse_args()

    
    if args.rows < 2 or args.cols < 2:
        output_error(
            algorithm="lattice_vqe",
            script_name="lattice_vqe.py",
            error_message="Lattice must be at least 2x2"
        )
    
    try:
        output = run_vqe(args)
        output_json(output)
        
    except Exception as e:
        output_error(
            algorithm="lattice_vqe",
            script_name="lattice_vqe.py",
            error_message=f"{type(e).__name__}: {str(e)}"
        )


if __name__ == '__main__':
    main()
