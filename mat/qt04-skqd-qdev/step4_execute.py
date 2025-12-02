"""
Step: Execute circuits on a backend and return results.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from qiskit_addon_sqd.counts import counts_to_arrays


def execute_circuits(
    circuits: list[QuantumCircuit],
    backend=None,
    shots: int = 1024
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Execute circuits on a backend and return bitstrings.
    
    Args:
        circuits: List of transpiled quantum circuits.
        backend: Target backend. Defaults to AerSimulator with density_matrix method.
        shots: Number of shots per circuit.
        
    Returns:
        Tuple of (bitstrings, probabilities, combined_counts).
    """
    if backend is None:
        backend = AerSimulator(method='density_matrix')
    
    sampler = SamplerV2(backend)
    
    # Run all circuits
    job = sampler.run(circuits)
    results = job.result()
    
    print(f"Executed {len(results)} circuits with {shots} shots each.")
    
    # Combine counts from all circuit results
    combined_counts = {}
    for pub_result in results:
        counts = pub_result.data.meas.get_counts()
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


if __name__ == "__main__":
    # Demo: execute a simple circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    bitstrings, probs, counts = run_step_execute([qc], shots=100)
    print(f"Demo complete: {counts}")
