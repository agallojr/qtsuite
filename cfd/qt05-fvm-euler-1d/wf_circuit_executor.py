"""
Circuit executor - runs on quantum backend venv site.
Loads circuit, transpiles, executes, and logs results.
We assume this code will be run in a venv with the necessary dependencies installed.
"""

#pylint: disable=wrong-import-position, invalid-name

import pickle
from datetime import datetime
import os
import sys
import traceback
from pathlib import Path
import json
import platform
from typing import cast

import numpy as np

from qiskit import qiskit, QuantumCircuit
from qiskit.qpy import load as qpy_load
from qiskit.qpy import dump as qpy_dump
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import Aer

def extract_quantum_solution(counts, matrix, classical_norm, caseId):
    """Extract quantum solution from measurement counts."""
    total_shots = sum(counts.values())
    n_qubits_matrix = int(np.log2(matrix.shape[0]))
    n_solution = 2 ** n_qubits_matrix
    quantum_solution = np.zeros(n_solution)

    print(f"[{caseId}] Extracting solution from {total_shots} shots, "
          f"n_qubits_matrix={n_qubits_matrix}")

    # HHL solution extraction: extract from last n_qubits_matrix bits
    for bitstring, count in counts.items():
        # Convert bitstring to state index (handle hex and binary formats)
        if bitstring.startswith('0x'):
            state_index = int(bitstring, 16)
        else:
            state_index = int(bitstring, 2)

        # Extract solution register bits (last n_qubits_matrix bits)
        solution_bits = state_index & ((1 << n_qubits_matrix) - 1)

        if solution_bits < n_solution:
            quantum_solution[solution_bits] += count / total_shots

    # Filter near-zero components
    quantum_solution[np.abs(quantum_solution) < 1e-10] = 0

    print(f"[{caseId}] Raw quantum solution: {quantum_solution}")

    # Normalize and scale to match classical solution norm
    if np.linalg.norm(quantum_solution) > 0:
        quantum_solution = (
            quantum_solution / np.linalg.norm(quantum_solution)
            * classical_norm
        )
    else:
        print(f"[{caseId}] WARNING: Zero norm quantum solution")
        quantum_solution = np.zeros(n_solution)

    print(f"[{caseId}] Normalized quantum solution: {quantum_solution}")

    return quantum_solution


def load_circuit_files(caseOutDir, iter_num, subiter_num):
    """Locate and load circuit and matrix data files for specific iteration and sub-iteration."""
    # Load specific iteration and sub-iteration circuit files
    circuit_qpy_path = caseOutDir / \
        f"hhl_circuit_iter{iter_num}_subiter{subiter_num}.qpy"
    circuit_pkl_path = caseOutDir / \
        f"hhl_circuit_iter{iter_num}_subiter{subiter_num}.pkl"

    if not circuit_qpy_path.exists():
        print(f"[ERROR] Circuit file not found: {circuit_qpy_path}")
        print(f"[ERROR] Available files: {list(caseOutDir.glob('hhl_circuit_iter*.qpy'))}")
        return None, None, None, None, None, None

    if not circuit_pkl_path.exists():
        print(f"[ERROR] Circuit pkl file not found: {circuit_pkl_path}")
        return None, None, None, None, None, None

    # Load circuit
    with open(circuit_qpy_path, "rb") as f:
        circuits = qpy_load(f)
        circuit = cast(QuantumCircuit, circuits[0] if isinstance(circuits, list) else circuits)

    # Ensure circuit has measurements
    circuit.measure_all()

    # Load matrix/vector data
    with open(circuit_pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
        matrix = pkl_data["matrix"]
        vector = pkl_data["vector"]
        matrix_herm = pkl_data["matrix_herm"]
        original_size = pkl_data.get("original_size", matrix.shape[0])

    return circuit, circuit_qpy_path, matrix, vector, matrix_herm, original_size


def debug_print(msg, caseId, log_file=None):
    """Print debug message to both console and log file."""

    # Format the message with timestamp and case ID
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    formatted_msg = f"[{timestamp}][{caseId}] {msg}"

    # Print to stderr (less likely to be buffered)
    print(formatted_msg)


def transpile_circuit(circuit, transpile_opt, quantum_backend, caseId):
    """Transpile circuit for target backend."""    
    # Set up log file in the current working directory
    log_dir = os.getcwd()
    log_file = os.path.join(log_dir, f'transpile_debug_{caseId}.log')

    # Clear previous log file if it exists
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"[WARNING] Could not remove old log file: {str(e)}", file=sys.stderr)

    debug_print("Starting circuit transpilation...", caseId, log_file)
    debug_print(f"Circuit qubits: {circuit.num_qubits}", caseId, log_file)
    debug_print(f"Circuit depth: {circuit.depth()}", caseId, log_file)
    debug_print(f"Circuit size: {circuit.size()}", caseId, log_file)
    debug_print(f"Circuit operations: {circuit.count_ops()}", caseId, log_file)

    try:
        # Get backend configuration and properties
        backend_info = str(quantum_backend)
        debug_print(f"Backend info: {backend_info}", caseId, log_file)

        debug_print(f"Creating pass manager with optimization level {transpile_opt}...", 
                   caseId, log_file)
        pass_manager = generate_preset_pass_manager(
            optimization_level=transpile_opt,
            backend=quantum_backend,  # Let Qiskit handle backend-specific optimizations
        )

        # Run the transpiler with progress tracking
        debug_print("Running pass manager...", caseId, log_file)
        import time
        start_time = time.time()

        # Wrap in a try-except to catch any transpilation errors
        try:
            transpiled_circuit = pass_manager.run(circuit)
            elapsed = time.time() - start_time
            debug_print(f"Transpilation completed in {elapsed:.2f} seconds",
                caseId, log_file)

            # Print basic info about the transpiled circuit
            debug_print("Transpiled circuit info:", caseId, log_file)
            debug_print(f"  - Qubits: {transpiled_circuit.num_qubits}", caseId, log_file)
            debug_print(f"  - Depth: {transpiled_circuit.depth()}", caseId, log_file)
            debug_print(f"  - Size: {transpiled_circuit.size()}", caseId, log_file)
            debug_print(f"  - Operations: {transpiled_circuit.count_ops()}", caseId, log_file)

            return transpiled_circuit

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Transpilation failed after {elapsed:.2f} " + \
                f"seconds: {type(e).__name__}: {str(e)}"
            debug_print(error_msg, caseId, log_file)
            
            # Log the full traceback
            error_trace = traceback.format_exc()
            debug_print(f"Error traceback:\n{error_trace}", caseId, log_file)
            
            # Try to save the circuit that caused the failure
            try:
                from qiskit import qpy
                failed_qc_path = f"failed_circuit_{caseId}.qpy"
                with open(failed_qc_path, 'wb') as fd:
                    qpy.dump(circuit, fd)
                debug_print(f"Saved failed circuit to {failed_qc_path}", caseId, log_file)
            except Exception as save_err:
                debug_print(f"Failed to save failed circuit: {str(save_err)}", caseId, log_file)
            
            raise  # Re-raise the exception
            
    except Exception as outer_e:
        error_msg = f"Fatal error in transpile_circuit: {str(outer_e)}"
        debug_print(error_msg, caseId, log_file)
        debug_print(f"Error traceback:\n{traceback.format_exc()}", caseId, log_file)
        raise


def execute_on_backend(transpiled_circuit, quantum_backend, shots, caseId):
    """Execute transpiled circuit on specified backend."""
    
    # Set up log file in the current working directory
    log_dir = os.getcwd()
    log_file = os.path.join(log_dir, f'execute_debug_{caseId}.log')
    
    def exec_debug(msg):
        """Helper function for debug logging"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        formatted_msg = f"[{timestamp}][{caseId}] {msg}"
        print(formatted_msg)
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{formatted_msg}\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {str(e)}", file=sys.stderr)

    exec_debug(f"Starting execution on backend: {quantum_backend}")
    exec_debug(f"Transpiled circuit info:")
    exec_debug(f"  - Qubits: {transpiled_circuit.num_qubits}")
    exec_debug(f"  - Depth: {transpiled_circuit.depth()}")
    exec_debug(f"  - Size: {transpiled_circuit.size()}")
    # Using Qiskit 1.x syntax for execution
    job = quantum_backend.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    print(f"[{caseId}] Execution complete")
    print(f"[{caseId}] Measurement counts: {counts}")

    return counts

def get_quantum_backend(backend_name, caseId):
    """Get a Qiskit Aer backend instance.
    
    Args:
        backend_name: Type of simulator to use. Can be one of:
            - 'density_matrix': Use Aer's density matrix simulator
            - 'statevector_simulator': Use Aer's statevector simulator
            - 'qasm_simulator': Use Aer's QASM simulator (default)
        caseId: Case ID for logging purposes
        
    Returns:
        A Qiskit Aer backend instance
    """
    try:
        if backend_name == 'density_matrix':
            backend = Aer.get_backend('aer_simulator_density_matrix')
            print(f"[{caseId}] Using Aer density matrix simulator")
        elif backend_name == 'statevector_simulator':
            backend = Aer.get_backend('aer_simulator_statevector')
            print(f"[{caseId}] Using Aer statevector simulator")
        elif backend_name == 'qasm_simulator':
            backend = Aer.get_backend('qasm_simulator')
            print(f"[{caseId}] Using Aer QASM simulator")
        else:
            print(f"[{caseId}] WARNING: Unknown backend '{backend_name}'. "
                  f"Using Aer QASM simulator instead.")
            backend = Aer.get_backend('qasm_simulator')
            
        return backend
    except Exception as e:
        print(f"[{caseId}] ERROR: Failed to initialize Aer backend: {str(e)}")
        raise


# ************************************************************************
# Helper functions for circuit execution workflow

def initialize_stats(circuit, iter_num, subiter_num, caseId, matrix, original_size):
    """Initialize statistics dictionary with pre-transpilation data."""
    condition_number = np.linalg.cond(matrix)
    orig_size = original_size if original_size is not None else matrix.shape[0]
    
    return {
        'iteration': iter_num,
        'sub_iteration': subiter_num,
        'case_id': caseId,
        'pre_transpile': {
            'num_qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'num_gates': circuit.size(),
            'gate_breakdown': dict(circuit.count_ops()),
            'estimated_runtime': (circuit.depth() * circuit.num_qubits *
                                2**circuit.num_qubits)
        },
        'matrix': {
            'original_size': int(orig_size),
            'padded_size': int(matrix.shape[0]),
            'shape': list(matrix.shape),
            'qubits_from_matrix': int(np.log2(matrix.shape[0])),
            'nonzeros': int(np.count_nonzero(matrix)),
            'density': float(np.count_nonzero(matrix) / matrix.size),
            'condition_number': float(condition_number)
        }
    }


def compute_classical_solutions(matrix, vector, caseId):
    """Compute classical reference solutions."""
    classical_solution = np.linalg.solve(matrix, vector/np.linalg.norm(vector))
    lu_solution_original = np.linalg.solve(matrix, vector)
    classical_norm = float(np.linalg.norm(classical_solution))
    
    print(f"[{caseId}] Classical solution computed, norm={classical_norm:.6f}")
    return classical_solution, lu_solution_original, classical_norm


def do_transpilation(circuit, transpile_opt, quantum_backend, caseId,
                    circuit_qpy_path, caseOutDir, start_times):
    """Transpile circuit and save result."""
    import time
    print(f"[{caseId}] Starting circuit transpilation (opt level {transpile_opt})...")
    infoWithSplits(f"[{caseId}] Starting circuit transpilation...", start_times)
    
    transpile_start = time.time()
    transpiled_circuit = transpile_circuit(
        circuit, transpile_opt, quantum_backend, caseId
    )
    transpile_time = time.time() - transpile_start
    
    # Capture post-transpilation statistics
    gate_counts = dict(transpiled_circuit.count_ops())
    
    # Categorize gates
    single_qubit_gates = sum(count for gate, count in gate_counts.items() 
                            if gate in ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg',
                                       'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 'p'])
    two_qubit_gates = sum(count for gate, count in gate_counts.items()
                         if gate in ['cx', 'cy', 'cz', 'ch', 'swap', 'iswap',
                                    'cswap', 'crx', 'cry', 'crz', 'cu', 'cp', 'cnot'])
    multi_qubit_gates = transpiled_circuit.size() - single_qubit_gates - two_qubit_gates
    
    # Calculate per-qubit statistics
    qubit_gate_counts = [0] * transpiled_circuit.num_qubits
    for instruction in transpiled_circuit.data:
        for qubit in instruction.qubits:
            qubit_gate_counts[transpiled_circuit.find_bit(qubit).index] += 1
    
    # Build qubit interaction matrix (connectivity heatmap)
    num_qubits = transpiled_circuit.num_qubits
    interaction_matrix = [[0 for _ in range(num_qubits)] for _ in range(num_qubits)]
    for instruction in transpiled_circuit.data:
        if len(instruction.qubits) == 2:  # Two-qubit gate
            q0 = transpiled_circuit.find_bit(instruction.qubits[0]).index
            q1 = transpiled_circuit.find_bit(instruction.qubits[1]).index
            interaction_matrix[q0][q1] += 1
            interaction_matrix[q1][q0] += 1
    
    # Circuit fingerprint: sample gate composition at different depths
    total_depth = transpiled_circuit.depth()
    num_samples = min(20, total_depth // 1000) if total_depth > 1000 else 10
    depth_samples = []
    if num_samples > 0:
        sample_interval = total_depth // num_samples
        current_depth = 0
        window_size = max(1000, sample_interval // 2)
        
        for sample_idx in range(num_samples):
            target_depth = sample_idx * sample_interval
            window_gates = {'single_qubit': 0, 'two_qubit': 0, 'multi_qubit': 0}
            window_gate_count = 0
            layer_depth = 0
            
            for instruction in transpiled_circuit.data:
                if layer_depth >= target_depth and layer_depth < target_depth + window_size:
                    window_gate_count += 1
                    num_qubits_in_gate = len(instruction.qubits)
                    if num_qubits_in_gate == 1:
                        window_gates['single_qubit'] += 1
                    elif num_qubits_in_gate == 2:
                        window_gates['two_qubit'] += 1
                    else:
                        window_gates['multi_qubit'] += 1
                layer_depth += 1
                if layer_depth >= target_depth + window_size:
                    break
            
            if window_gate_count > 0:
                depth_samples.append({
                    'depth': target_depth,
                    'depth_pct': 100 * target_depth / total_depth,
                    'single_pct': 100 * window_gates['single_qubit'] / window_gate_count,
                    'two_pct': 100 * window_gates['two_qubit'] / window_gate_count,
                    'multi_pct': 100 * window_gates['multi_qubit'] / window_gate_count
                })
    
    stats = {
        'num_qubits': transpiled_circuit.num_qubits,
        'depth': transpiled_circuit.depth(),
        'num_gates': transpiled_circuit.size(),
        'gate_breakdown': gate_counts,
        'gate_categories': {
            'single_qubit': single_qubit_gates,
            'two_qubit': two_qubit_gates,
            'multi_qubit': multi_qubit_gates
        },
        'qubit_utilization': {
            'min_gates': min(qubit_gate_counts) if qubit_gate_counts else 0,
            'max_gates': max(qubit_gate_counts) if qubit_gate_counts else 0,
            'avg_gates': sum(qubit_gate_counts) / len(qubit_gate_counts) if qubit_gate_counts else 0,
            'per_qubit': qubit_gate_counts
        },
        'circuit_metrics': {
            'depth_to_width_ratio': transpiled_circuit.depth() / transpiled_circuit.num_qubits if transpiled_circuit.num_qubits > 0 else 0,
            'gates_per_qubit': transpiled_circuit.size() / transpiled_circuit.num_qubits if transpiled_circuit.num_qubits > 0 else 0,
            'parallelism_factor': transpiled_circuit.size() / transpiled_circuit.depth() if transpiled_circuit.depth() > 0 else 0
        },
        'qubit_interaction_matrix': interaction_matrix,
        'depth_fingerprint': depth_samples,
        'estimated_runtime': (transpiled_circuit.depth() *
                            transpiled_circuit.num_qubits *
                            2**transpiled_circuit.num_qubits),
        'transpile_time_seconds': transpile_time
    }
    
    infoWithSplits(
        f"[{caseId}] Transpilation done: {transpiled_circuit.num_qubits} qubits, "
        f"depth={transpiled_circuit.depth()}, gates={transpiled_circuit.size()}",
        start_times
    )
    
    # Save transpiled circuit
    base_name = circuit_qpy_path.stem
    transpiled_path = caseOutDir / f"{base_name}_transpiled.qpy"
    with open(transpiled_path, "wb") as f:
        qpy_dump(transpiled_circuit, f)
    print(f"[{caseId}] Saved transpiled circuit to {transpiled_path}")
    
    return transpiled_circuit, stats


def do_execution(transpiled_circuit, quantum_backend, shots, matrix,
                classical_norm, caseId, start_times):
    """Execute circuit and extract quantum solution."""
    import time
    infoWithSplits(f"[{caseId}] Starting circuit execution...", start_times)
    
    exec_start = time.time()
    counts = execute_on_backend(
        transpiled_circuit, quantum_backend, shots, caseId
    )
    exec_time = time.time() - exec_start
    
    if counts is None:
        print(f"[{caseId}] ERROR: Circuit execution returned no counts")
        return None, None
    
    infoWithSplits(
        f"[{caseId}] Execution complete, {len(counts)} outcomes",
        start_times
    )
    
    # Extract quantum solution
    quantum_solution = extract_quantum_solution(
        counts, matrix, classical_norm, caseId
    )
    if quantum_solution is None:
        print(f"[{caseId}] ERROR: Failed to extract quantum solution")
        return None, None
    
    infoWithSplits(
        f"[{caseId}] Quantum solution extracted, shape={quantum_solution.shape}",
        start_times
    )
    
    return quantum_solution, exec_time


def compute_solution_comparisons(quantum_solution, classical_solution,
                                lu_solution_original, original_size, caseId):
    """Compute metrics comparing quantum, classical, and LU solutions."""
    # Truncate to original size if needed
    if quantum_solution.shape[0] > original_size:
        quantum_solution = quantum_solution[:original_size]
        classical_solution = classical_solution[:original_size]
        lu_solution_original = lu_solution_original[:original_size]
    
    # Calculate norms
    quantum_norm = float(np.linalg.norm(quantum_solution))
    classical_norm = float(np.linalg.norm(classical_solution))
    lu_norm = float(np.linalg.norm(lu_solution_original))
    
    # Normalize solutions
    quantum_normalized = (quantum_solution / quantum_norm
                         if quantum_norm > 0 else quantum_solution)
    classical_normalized = (classical_solution / classical_norm
                           if classical_norm > 0 else classical_solution)
    lu_normalized = (lu_solution_original / lu_norm
                    if lu_norm > 0 else lu_solution_original)
    
    # Calculate pairwise metrics
    def calc_metrics(sol1, sol2, name1, name2):
        abs_diff = float(np.linalg.norm(sol1 - sol2))
        norm2 = np.linalg.norm(sol2)
        rel_diff = abs_diff / norm2 if norm2 > 0 else None
        dot_prod = np.dot(sol1, sol2) / (np.linalg.norm(sol1) * norm2)
        angle_deg = float(np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0))))
        return {
            'absolute_difference': abs_diff,
            'relative_difference': float(rel_diff) if rel_diff else None,
            'angle_degrees': angle_deg if not np.isnan(angle_deg) else None,
            f'{name1}_norm': float(np.linalg.norm(sol1)),
            f'{name2}_norm': float(np.linalg.norm(sol2))
        }
    
    return {
        'solutions': {
            'quantum': {
                'vector': quantum_solution.tolist(),
                'normalized_vector': quantum_normalized.tolist(),
                'norm': quantum_norm
            },
            'classical': {
                'vector': classical_solution.tolist(),
                'normalized_vector': classical_normalized.tolist(),
                'norm': classical_norm
            },
            'lu': {
                'vector': lu_solution_original.tolist(),
                'normalized_vector': lu_normalized.tolist(),
                'norm': lu_norm
            }
        },
        'comparisons': {
            'quantum_vs_classical': calc_metrics(
                quantum_solution, classical_solution, 'quantum', 'classical'
            ),
            'quantum_vs_lu': calc_metrics(
                quantum_solution, lu_solution_original, 'quantum', 'lu'
            ),
            'classical_vs_lu': calc_metrics(
                classical_solution, lu_solution_original, 'classical', 'lu'
            ),
            'quantum_normalized_vs_classical': calc_metrics(
                quantum_normalized, classical_normalized, 'quantum', 'classical'
            ),
            'quantum_normalized_vs_lu': calc_metrics(
                quantum_normalized, lu_normalized, 'quantum', 'lu'
            ),
            'classical_normalized_vs_lu': calc_metrics(
                classical_normalized, lu_normalized, 'classical', 'lu'
            )
        }
    }


def save_results(stats, quantum_solution, caseOutDir, iter_num, subiter_num, caseId):
    """Save statistics and quantum solution to files."""
    # Save stats
    stats_path = caseOutDir / f"circuit_stats_iter{iter_num}_subiter{subiter_num}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[{caseId}] Circuit statistics saved to {stats_path}")
    
    if quantum_solution is None:
        return
    
    # Save quantum solution
    solution_path = caseOutDir / f"quantum_solution_iter{iter_num}_subiter{subiter_num}.npy"
    
    if not isinstance(quantum_solution, np.ndarray):
        print(f"[{caseId}] ERROR: Quantum solution is not a numpy array")
        return
    
    if np.any(np.isnan(quantum_solution)) or np.any(np.isinf(quantum_solution)):
        print(f"[{caseId}] ERROR: Quantum solution contains invalid values")
        return
    
    np.save(solution_path, quantum_solution)
    
    if not solution_path.exists():
        print(f"[{caseId}] ERROR: Failed to save solution file")
        return
    
    file_size = solution_path.stat().st_size
    print(f"[{caseId}] Solution saved: {solution_path.name} ({file_size} bytes)")


# ************************************************************************
# Main routine

def execute_circuit(casedir: str, iter_num: int, subiter_num: int = 0,
                    run_circuit: bool = True):
    """
    Orchestrate circuit transpilation and optional execution.
    
    Args:
        casedir: Case directory path
        iter_num: Outer iteration number
        subiter_num: Inner iteration number
        run_circuit: If True, execute circuit; if False, only transpile
    
    Returns:
        Quantum solution vector if run_circuit=True, None otherwise
    """
    start_times: list[float] = []
    
    # Load context and files
    context_file = Path(casedir) / "wf_context.json"
    with open(context_file, 'r', encoding='utf-8') as f:
        context = json.load(f)
    
    caseId = context['case_id']
    caseOutDir = Path(context['casedir'])
    
    infoWithSplits(
        f"[{caseId}] Starting circuit workflow for iter={iter_num}, subiter={subiter_num}",
        start_times
    )
    
    # Load circuit and matrix data
    circuit, circuit_qpy_path, matrix, vector, matrix_herm, original_size = \
        load_circuit_files(caseOutDir, iter_num, subiter_num)
    if circuit is None:
        return None
    
    # Load vector_herm from pkl file
    circuit_pkl_path = caseOutDir / f"hhl_circuit_iter{iter_num}_subiter{subiter_num}.pkl"
    vector_herm = None
    if circuit_pkl_path.exists():
        try:
            with open(circuit_pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
                vector_herm = pkl_data.get("vector_herm")
        except Exception:
            pass
    
    matrix = cast(np.ndarray, matrix)
    vector = cast(np.ndarray, vector)
    
    # Initialize statistics
    stats = initialize_stats(circuit, iter_num, subiter_num, caseId, matrix, original_size)
    
    # Compute classical solutions
    classical_solution, lu_solution_original, classical_norm = \
        compute_classical_solutions(matrix, vector, caseId)
    
    # Get quantum parameters
    case_args = context.get('case_args', {})
    backend_name = case_args.get('quantum_backend', 'automatic_sim_aer')
    transpile_opt = case_args.get('quantum_transpile_opt', 1)
    shots = case_args.get('quantum_shots', 1024)
    quantum_backend = get_quantum_backend(backend_name, caseId)
    
    stats['execution_params'] = {
        'backend': backend_name,
        'backend_type': type(quantum_backend).__name__,
        'transpile_optimization': transpile_opt,
        'shots': shots
    }
    stats['classical_solution'] = {
        'norm': classical_norm,
        'vector': classical_solution.tolist()
    }
    
    # Compute padded matrix metrics (this is A_pow2, not original)
    padded_cond = None
    padded_det = None
    print(f"[{caseId}] Computing padded matrix metrics: shape={matrix.shape}")
    try:
        padded_cond = float(np.linalg.cond(matrix))
        padded_det = float(np.linalg.det(matrix))
        print(f"[{caseId}] Padded cond={padded_cond:.6e}, det={padded_det:.6e}")
    except Exception as e:
        print(f"[{caseId}] Error computing padded metrics: {e}")
    
    stats['matrix_padded'] = {
        'matrix': matrix.tolist(),
        'shape': list(matrix.shape),
        'condition_number': padded_cond,
        'determinant': padded_det
    }
    
    # Compute Hermitian matrix metrics
    herm_cond = None
    herm_det = None
    if matrix_herm is not None:
        print(f"[{caseId}] Computing Hermitian metrics: shape={matrix_herm.shape}")
        print(f"[{caseId}] Padded shape={matrix.shape}, Hermitian shape={matrix_herm.shape}")
        print(f"[{caseId}] Are they the same matrix? {np.array_equal(matrix, matrix_herm[:matrix.shape[0], :matrix.shape[1]])}")
        try:
            herm_cond = float(np.linalg.cond(matrix_herm))
            herm_det = float(np.linalg.det(matrix_herm))
            print(f"[{caseId}] Hermitian cond={herm_cond:.6e}, det={herm_det:.6e}")
        except Exception as e:
            print(f"[{caseId}] Error computing Hermitian metrics: {e}")
    else:
        print(f"[{caseId}] WARNING: matrix_herm is None!")
    
    stats['matrix_hermitian'] = {
        'matrix': matrix_herm.tolist() if matrix_herm is not None else None,
        'shape': list(matrix_herm.shape) if matrix_herm is not None else None,
        'condition_number': herm_cond,
        'determinant': herm_det
    }
    stats['vector_hermitian'] = {
        'vector': vector_herm.tolist() if vector_herm is not None else None,
        'shape': list(vector_herm.shape) if vector_herm is not None else None
    }
    stats['nelem'] = case_args.get('nelem', 'N/A')
    
    # Read generation time if available
    gen_time_file = caseOutDir / f'gen_time_iter{iter_num}_subiter{subiter_num}.json'
    gen_time = None
    if gen_time_file.exists():
        try:
            with open(gen_time_file, 'r', encoding='utf-8') as f:
                gen_data = json.load(f)
                gen_time = gen_data.get('generation_time_seconds')
        except Exception:
            pass
    
    # Transpile circuit
    try:
        transpiled_circuit, post_transpile_stats = do_transpilation(
            circuit, transpile_opt, quantum_backend, caseId,
            circuit_qpy_path, caseOutDir, start_times
        )
        stats['post_transpile'] = post_transpile_stats
        if gen_time is not None:
            stats['post_transpile']['generation_time_seconds'] = gen_time
    except Exception as e:
        print(f"[{caseId}] ERROR: Transpilation failed: {e}")
        return None
    
    # Check if we should execute
    if not run_circuit:
        print(f"[{caseId}] Transpilation complete, execution skipped")
        save_results(stats, None, caseOutDir, iter_num, subiter_num, caseId)
        return None

    # Execute circuit and extract solution
    try:
        quantum_solution, exec_time = do_execution(
            transpiled_circuit, quantum_backend, shots, matrix,
            classical_norm, caseId, start_times
        )
        if quantum_solution is None:
            return None
        
        # Add execution time to stats
        stats['post_transpile']['execution_time_seconds'] = exec_time
        
        # Compute solution comparisons
        comparison_data = compute_solution_comparisons(
            quantum_solution, classical_solution, lu_solution_original,
            original_size, caseId
        )
        stats.update(comparison_data)
        
        # Log solution metrics
        qc_diff = stats['comparisons']['quantum_vs_classical']
        rel_err = qc_diff['relative_difference']
        rel_err_str = f"{rel_err:.4e}" if rel_err is not None else "N/A"
        infoWithSplits(
            f"[{caseId}] Solution relative error: {rel_err_str}",
            start_times
        )
        
        # Add timing info
        stats['timing'] = {
            'phases': {},
            'total_elapsed': (start_times[-1] - start_times[0]
                            if len(start_times) >= 2 else None),
            'raw_timestamps': start_times
        }
        
        # Save results
        save_results(stats, quantum_solution, caseOutDir, iter_num, subiter_num, caseId)
        return quantum_solution

    except Exception as e:
        error_msg = f"Circuit execution failed: {str(e)}\n{traceback.format_exc()}"
        
        # Log to file with detailed error information
        error_log_path = Path(caseOutDir) / f"error_iter{iter_num}_subiter{subiter_num}.log"
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Error in circuit execution:\n{error_msg}\n\n")
            f.write("Current working directory: " + str(Path.cwd()) + "\n")
            f.write(f"Case directory: {caseOutDir}\n")
            f.write(f"Case ID: {caseId}\n")
            f.write(f"Iteration: {iter_num}, Subiteration: {subiter_num}\n\n")
            
            # Write environment information
            f.write("Environment:\n")
            f.write(f"Python: {platform.python_version()}\n")
            f.write(f"Qiskit: {qiskit.__version__}\n")
            f.write(f"Numpy: {np.__version__}\n\n")
            
            # Write the full traceback
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        
        # Log error to console and log file
        print(f"[{caseId}] ERROR: {error_msg}")
        print(f"[{caseId}] ERROR: Detailed error log saved to: {error_log_path}")
        
        # Re-raise the exception to ensure the workflow is aware of the failure
        error_msg = (
            f"[{caseId}] Circuit execution failed. "
            f"See {error_log_path} for details."
        )
        raise RuntimeError(error_msg) from e

    return quantum_solution


# ******************************************************************************

# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-casedir", required=True, 
#         help="Case directory")
#     parser.add_argument("-iter", type=int, required=True, 
#         help="Outer iteration number")
#     parser.add_argument("-subiter", type=int, default=0, 
#         help="Inner (Newton) iteration number")
#     args = parser.parse_args()

#     execute_circuit(args.casedir, args.iter, args.subiter)
