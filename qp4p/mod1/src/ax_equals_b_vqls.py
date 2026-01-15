"""
Solve Ax=b using VQLS (Variational Quantum Linear Solver)

VQLS is a hybrid quantum-classical algorithm that uses a variational ansatz
to find the solution to a linear system. Unlike HHL, it's designed for NISQ devices.

Key concepts:
- The ansatz encodes |x⟩ as a parameterized quantum state
- Classical optimizer minimizes cost = 1 - |⟨b|A|x⟩|² / (||Ax||² ||b||²)
- When cost → 0, A|x⟩ is parallel to |b⟩, meaning x solves Ax=b

Phase handling:
- VQLS finds x such that A|x⟩ ∝ |b⟩ (proportional, not equal)
- The solution is determined up to a global phase/sign
- Fidelity = |⟨x_classical|x_quantum⟩|² accounts for this by using absolute overlap
- For real systems, this means the solution may have opposite sign but same direction

"""

import argparse
import json
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Statevector
from qp4p_args import add_noise_args, add_backend_args
from qp4p_output import create_standardized_output
from qp4p_circuit import BASIS_GATES, get_fake_backend
from qp4p_linear_system import add_linear_system_args, get_linear_system


# Matrix parsing and validation moved to qp4p_linear_system helper


def build_ansatz(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Build a variational ansatz for VQLS."""
    return real_amplitudes(num_qubits, reps=reps)


def prepare_b_state(b: np.ndarray) -> QuantumCircuit:
    """Prepare the |b⟩ state from vector b."""
    b_norm = b / np.linalg.norm(b)
    num_qubits = int(np.log2(len(b)))
    qc = QuantumCircuit(num_qubits)
    qc.initialize(b_norm, range(num_qubits))
    return qc


def vqls_cost_function(params, ansatz, A, b_state, num_qubits):  # pylint: disable=unused-argument
    """
    Compute the VQLS cost function.
    
    Cost = 1 - |⟨b|A|x(θ)⟩|² / (⟨x(θ)|A†A|x(θ)⟩ * ⟨b|b⟩)
    
    This measures how well A|x⟩ aligns with |b⟩.
    """
    # Bind parameters to ansatz
    bound_ansatz = ansatz.assign_parameters(params)
    
    # Get |x(θ)⟩ statevector
    x_state = Statevector.from_instruction(bound_ansatz)
    x_vec = x_state.data
    
    # Get |b⟩ statevector
    b_sv = Statevector.from_instruction(b_state)
    b_vec = b_sv.data
    
    # Compute A|x⟩
    Ax = A @ x_vec
    
    # Compute cost: we want A|x⟩ ∝ |b⟩
    # Cost = 1 - |⟨b|Ax⟩|² / (||Ax||² * ||b||²)
    numerator = np.abs(np.vdot(b_vec, Ax)) ** 2
    denominator = np.linalg.norm(Ax) ** 2 * np.linalg.norm(b_vec) ** 2
    
    if denominator < 1e-10:
        return 1.0
    
    cost = 1.0 - numerator / denominator
    return float(np.real(cost))


def extract_solution(params, ansatz, A, b):
    """Extract the solution vector from optimized ansatz."""
    bound_ansatz = ansatz.assign_parameters(params)
    x_state = Statevector.from_instruction(bound_ansatz)
    x_vec = x_state.data
    
    # The ansatz gives |x⟩ normalized. We need to scale it.
    # From Ax = b, we have x = A^{-1}b
    # The ansatz gives x/||x||, so we need to find the scale factor
    
    # Compute A|x⟩
    Ax = A @ x_vec
    
    # Find scale factor: Ax should equal b (up to normalization)
    b_norm = b / np.linalg.norm(b)
    
    # Scale factor = ||b|| / ||Ax|| * sign
    scale = np.linalg.norm(b) / np.linalg.norm(Ax)
    
    # Get the solution
    solution = np.real(x_vec * scale)
    
    return solution


def solve_vqls(A: np.ndarray, b: np.ndarray, maxiter: int = 200, reps: int = 3) -> tuple:
    """
    Solve Ax=b using VQLS.
    
    Returns:
        (solution, ansatz, optimal_params, optimization_result)
    """
    n = len(b)
    num_qubits = int(np.log2(n))
    
    # Build ansatz
    ansatz = build_ansatz(num_qubits, reps=reps)
    num_params = ansatz.num_parameters
    
    # Prepare |b⟩ state
    b_state = prepare_b_state(b)
    
    # Initial parameters
    initial_params = np.random.uniform(-np.pi, np.pi, num_params)
    
    # Optimize
    result = minimize(
        vqls_cost_function,
        initial_params,
        args=(ansatz, A, b_state, num_qubits),
        method='COBYLA',
        options={'maxiter': maxiter, 'rhobeg': 0.5}
    )
    
    # Extract solution
    solution = extract_solution(result.x, ansatz, A, b)
    
    return solution, ansatz, result.x, result


def compute_fidelity(classical_x: np.ndarray, quantum_x: np.ndarray) -> float:
    """Compute fidelity between classical and quantum solutions."""
    c_norm = classical_x / np.linalg.norm(classical_x)
    q_norm = quantum_x / np.linalg.norm(quantum_x)
    
    fidelity = np.abs(np.dot(c_norm, q_norm)) ** 2
    return float(fidelity)


# *****************************************************************************
# main

# Random system generation moved to qp4p_linear_system helper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Ax=b using VQLS (Variational Quantum Linear Solver)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ax_equals_b_vqls.py --size 4
  python ax_equals_b_vqls.py --size 4 --seed 42
  python ax_equals_b_vqls.py --size 4 --tridiag
  python ax_equals_b_vqls.py --matrix "[[2,1],[1,2]]" --vector "[1,0]"
""")
    add_linear_system_args(parser)
    parser.add_argument("--maxiter", type=int, default=200,
                        help="Max optimizer iterations (default: 200)")
    parser.add_argument("--reps", type=int, default=3,
                        help="Ansatz repetitions/depth (default: 3)")
    add_noise_args(parser)
    add_backend_args(parser)
    args = parser.parse_args()

    # 1. Get the linear system (handles random, tridiagonal, or explicit)
    # Default to size=4, seed=42 if nothing specified
    if args.size is None and args.matrix is None:
        args.size = 4
        args.seed = 42
    
    A_orig, b_orig, sys_metadata = get_linear_system(args, require_power_of_2=True)
    original_size = sys_metadata['original_size']
    
    # Classical solution (use original unpadded system)
    A_unpadded = A_orig[:original_size, :original_size]
    b_unpadded = b_orig[:original_size]
    classical_solution = np.linalg.solve(A_unpadded, b_unpadded)
    
    # 2. Use padded system for quantum
    A, b = A_orig, b_orig
    n = A.shape[0]
    padded_size = sys_metadata.get('padded_size', n)
    num_qubits = int(np.log2(n))
    
    # 3. Solve with VQLS
    quantum_solution_full, ansatz, optimal_params, opt_result = solve_vqls(A, b, args.maxiter, args.reps)
    
    # Extract original components
    quantum_solution = quantum_solution_full[:original_size]
    
    # 4. Normalize solutions for comparison
    classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)
    quantum_solution_normalized = quantum_solution / np.linalg.norm(quantum_solution)
    
    # Compute metrics on normalized solutions
    fidelity = compute_fidelity(classical_solution, quantum_solution)
    l2_error = float(np.linalg.norm(quantum_solution_normalized - classical_solution_normalized))
    
    # 5. Transpile the bound ansatz for backend stats
    # Bind optimal parameters to get the actual circuit that would be executed
    bound_ansatz = ansatz.assign_parameters(optimal_params)
    
    if args.backend:
        backend = get_fake_backend(args.backend)
        transpiled = transpile(bound_ansatz, backend=backend, optimization_level=2)
        backend_info = {"name": args.backend, "num_qubits": int(backend.num_qubits)}
    else:
        transpiled = transpile(bound_ansatz, basis_gates=BASIS_GATES, optimization_level=2)
        backend_info = None
    
    transpiled_stats = {
        "depth": int(transpiled.depth()),
        "gate_counts": {k: int(v) for k, v in transpiled.count_ops().items()},
        "num_qubits": int(transpiled.num_qubits)
    }
    
    # 6. Build results
    output = create_standardized_output(
        algorithm="vqls",
        script_name="ax_equals_b_vqls.py",
        problem={
            "matrix": A_orig.tolist(),
            "rhs": b_orig.tolist(),
            "dimension": original_size,
            "condition_number": float(np.linalg.cond(A_orig))
        },
        config={
            "ansatz_reps": args.reps,
            "maxiter": args.maxiter,
            "t1": args.t1,
            "t2": args.t2,
            "backend": args.backend
        },
        results={
            "classical_solution": classical_solution.tolist(),
            "classical_solution_normalized": classical_solution_normalized.tolist(),
            "quantum_solution": quantum_solution.tolist(),
            "quantum_solution_normalized": quantum_solution_normalized.tolist()
        },
        metrics={
            "fidelity": float(fidelity),
            "l2_error": l2_error,
            "optimization_iterations": int(opt_result.nfev),
            "optimization_cost": round(float(opt_result.fun), 6),
            "optimization_success": bool(opt_result.success) if hasattr(opt_result, 'success') else None
        },
        circuit_info={
            "num_qubits": int(ansatz.num_qubits),
            "depth": int(ansatz.depth()),
            "num_parameters": int(ansatz.num_parameters),
            "transpiled_stats": transpiled_stats
        },
        backend_info=backend_info
    )
    
    # Print JSON to stdout
    print(json.dumps(output, indent=2))
