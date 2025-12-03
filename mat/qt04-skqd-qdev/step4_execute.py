"""
Step: Execute circuits on a backend and return results.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit_addon_sqd.counts import counts_to_arrays


def execute_circuits(
    circuits: list[QuantumCircuit],
    backend=None,
    shots: int = 1024
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Execute circuits on a backend and return bitstrings.
    
    Args:
        circuits: List of transpiled quantum circuits.
        backend: Target backend. Defaults to AerSimulator.
        shots: Number of shots per circuit.
        
    Returns:
        Tuple of (bitstrings, probabilities, combined_counts).
    """
    if backend is None:
        backend = AerSimulator(method='automatic')
    
    # Run circuits directly on backend
    job = backend.run(circuits, shots=shots)
    result = job.result()
    
    print(f"Executed {len(circuits)} circuits with {shots} shots each.")
    
    # Combine counts from all circuit results
    combined_counts = {}
    for i in range(len(circuits)):
        counts = result.get_counts(i)
        for bitstring, count in counts.items():
            combined_counts[bitstring] = combined_counts.get(bitstring, 0) + count
    
    # Convert to arrays
    bitstrings, probabilities = counts_to_arrays(combined_counts)
    
    print(f"Collected {len(combined_counts)} unique bitstrings.")
    print(f"Bitstrings shape: {bitstrings.shape}")
    
    return bitstrings, probabilities, combined_counts


def run_step_execute(
    circuits: list[QuantumCircuit],
    shots: int = 1024
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run execution step with default Aer density_matrix backend.
    
    Args:
        circuits: Transpiled circuits to execute.
        shots: Number of shots per circuit.
        
    Returns:
        Tuple of (bitstrings, probabilities, combined_counts).
    """
    return execute_circuits(circuits, shots=shots)

