"""
VQE helper functions providing unified interface for variational quantum eigensolver.

Supports two methods:
- 'qiskit': Uses Qiskit's VQE class with EstimatorV2
- 'manual': Manual SPSA optimization with shots-based estimation
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
from qiskit_aer.primitives import EstimatorV2

from qp4p_circuit import build_noise_model, run_estimator
from qp4p_opt import spsa_optimize


def run_vqe_optimization(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit = None,
    optimizer: str = "COBYLA",
    method: str = "qiskit",
    maxiter: int = 100,
    shots: int = None,
    t1: float = None,
    t2: float = None,
    backend: str = None,
    coupling_map: str = "default",
    seed: int = 42,
    num_attempts: int = 1,
    callback: callable = None
) -> dict:
    """
    Run VQE optimization with unified interface.
    
    Args:
        hamiltonian: Hamiltonian as SparsePauliOp
        ansatz: Ansatz circuit (if None, creates TwoLocal for qiskit or EfficientSU2 for manual)
        optimizer: Optimizer name ('COBYLA', 'SLSQP', 'SPSA')
        method: 'qiskit' (uses VQE class) or 'manual' (SPSA with multiple attempts)
        maxiter: Maximum optimizer iterations
        shots: Number of shots (None = statevector for qiskit method)
        t1: T1 relaxation time in microseconds
        t2: T2 dephasing time in microseconds
        backend: Fake backend name (e.g., 'manila', 'jakarta')
        coupling_map: "default" (backend's coupling) or "all-to-all" (full connectivity)
        seed: Random seed
        num_attempts: Number of optimization attempts (manual method only)
        callback: Callback function for energy history (qiskit method only)
    
    Returns:
        dict with:
            - energy: Ground state energy
            - optimal_params: Optimal parameters
            - evaluations: Number of function evaluations
            - energy_history: Energy convergence history (if available)
            - ansatz_info: Ansatz metadata
            - method: Method used
            - attempts: Attempt details (manual method only)
    """
    num_qubits = hamiltonian.num_qubits
    
    # Create default ansatz if not provided
    if ansatz is None:
        if method == "qiskit":
            ansatz = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cx',
                            entanglement='linear', reps=2)
        else:  # manual
            ansatz = EfficientSU2(num_qubits, reps=2, entanglement='linear')
    
    # Decompose ansatz for compatibility
    ansatz = ansatz.decompose().decompose()
    
    if method == "qiskit":
        return _run_vqe_qiskit(
            hamiltonian, ansatz, optimizer, maxiter, 
            t1, t2, backend, coupling_map, seed, callback
        )
    elif method == "manual":
        return _run_vqe_manual(
            hamiltonian, ansatz, maxiter, shots or 1024,
            t1, t2, backend, coupling_map, num_attempts
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'qiskit' or 'manual'")


def _run_vqe_qiskit(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer_name: str,
    maxiter: int,
    t1: float,
    t2: float,
    backend: str,
    coupling_map: str,
    seed: int,
    callback: callable
) -> dict:
    """Run VQE using Qiskit's VQE class with EstimatorV2."""
    
    # Build noise model
    noise_model = None
    fake_backend = None
    coupling_map_obj = None
    if backend or (t1 is not None and t2 is not None):
        noise_model, fake_backend, coupling_map_obj = build_noise_model(
            t1=t1, t2=t2, backend=backend, coupling_map=coupling_map)
    
    # Create estimator
    if noise_model is not None:
        estimator = EstimatorV2(
            options={
                "default_precision": 0.01,
                "backend_options": {
                    "method": "statevector",
                    "noise_model": noise_model,
                }
            }
        )
    else:
        estimator = EstimatorV2(
            options={"default_precision": 0.01}
        )
    
    # Create optimizer
    if optimizer_name == "COBYLA":
        optimizer = COBYLA(maxiter=maxiter)
    elif optimizer_name == "SLSQP":
        optimizer = SLSQP(maxiter=maxiter)
    elif optimizer_name == "SPSA":
        optimizer = SPSA(maxiter=maxiter)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Track energy history
    energy_history = []
    
    def internal_callback(eval_count, params, mean, std):
        energy_history.append(float(mean))
        if callback:
            callback(eval_count, params, mean, std)
    
    # Set random seed
    algorithm_globals.random_seed = seed
    
    # Run VQE
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, 
              callback=internal_callback)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    
    # Convert optimal_parameters dict to JSON-serializable format
    optimal_params = None
    if hasattr(result, 'optimal_parameters') and result.optimal_parameters is not None:
        # Convert ParameterVectorElement keys to strings
        optimal_params = {str(k): float(v) for k, v in result.optimal_parameters.items()}
    
    return {
        "energy": float(result.eigenvalue),
        "optimal_params": optimal_params,
        "evaluations": int(result.cost_function_evals),
        "energy_history": energy_history,
        "ansatz_info": {
            "num_qubits": ansatz.num_qubits,
            "num_parameters": ansatz.num_parameters,
            "depth": ansatz.depth()
        },
        "method": "qiskit",
        "optimizer": optimizer_name,
        "converged": True
    }


def _run_vqe_manual(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    maxiter: int,
    shots: int,
    t1: float,
    t2: float,
    backend: str,
    coupling_map: str,
    num_attempts: int
) -> dict:
    """Run VQE using manual SPSA optimization with multiple attempts."""
    
    # Transpile ansatz
    transpiled_ansatz = transpile(ansatz, basis_gates=['cx', 'h', 'x', 'y', 'z', 'rz', 'ry', 'rx'])
    
    eval_count = [0]
    
    def vqe_cost(params):
        """Compute <ψ(θ)|H|ψ(θ)> using shots-based Estimator"""
        bound_circuit = transpiled_ansatz.assign_parameters(params)
        energy = run_estimator(bound_circuit, hamiltonian, shots=shots, 
                               t1=t1, t2=t2, backend=backend, coupling_map=coupling_map)
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
        "energy": float(best_result["fun"]),
        "optimal_params": best_result["x"].tolist(),
        "evaluations": sum(a["evaluations"] for a in all_attempts),
        "energy_history": None,  # Not tracked in manual method
        "ansatz_info": {
            "num_qubits": ansatz.num_qubits,
            "num_parameters": ansatz.num_parameters,
            "depth": transpiled_ansatz.depth()
        },
        "method": "manual",
        "optimizer": "SPSA",
        "converged": True,
        "attempts": all_attempts,
        "num_attempts": num_attempts
    }
