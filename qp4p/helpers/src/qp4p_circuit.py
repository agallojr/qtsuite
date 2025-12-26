"""
Helper functions for running quantum circuits.
"""
import math
import json
import numpy as np
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_aer.primitives import EstimatorV2
import matplotlib.pyplot as plt

# Basis gates supported by Aer simulator
BASIS_GATES = ['cx', 'id', 'rz', 'sx', 'x']


def build_noise_model(t1: float = None, t2: float = None) -> NoiseModel:
    """
    Build a thermal relaxation noise model.
    
    Args:
        t1: T1 relaxation time in microseconds (energy decay). None = no noise.
        t2: T2 dephasing time in microseconds. Must be <= 2*T1.
    
    Returns:
        NoiseModel or None if no noise parameters specified.
    
    Typical values for superconducting qubits: T1 ~ 50-150 µs, T2 ~ 50-120 µs.
    """
    if t1 is not None and t1 <= 0:
        t1 = None
    if t2 is not None and t2 <= 0:
        t2 = None
    
    if t1 is None or t2 is None:
        return None
    
    noise_model = NoiseModel()
    
    # Gate times in microseconds (typical for superconducting qubits)
    gate_time_1q = 0.05  # 50 ns for single-qubit gates
    gate_time_2q = 0.3   # 300 ns for two-qubit gates
    
    error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q))
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'y', 'z', 'ry', 'rz', 'rx'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    return noise_model


def run_circuit(qc: QuantumCircuit, shots: int = 1024, 
                t1: float = None, t2: float = None) -> dict:
    """
    Run a circuit on the Aer simulator and return the counts.
    
    Args:
        qc: The quantum circuit to run.
        shots: Number of shots.
        t1: T1 relaxation time in microseconds (energy decay). None = no noise.
        t2: T2 dephasing time in microseconds (dephasing). Must be <= 2*T1.
    
    Typical values for superconducting qubits: T1 ~ 50-150 µs, T2 ~ 50-120 µs.
    Longer is better.
    """
    noise_model = build_noise_model(t1, t2)
    simulator = AerSimulator()
    
    if noise_model is not None:
        result = simulator.run(qc, shots=shots, noise_model=noise_model).result()
    else:
        result = simulator.run(qc, shots=shots).result()
    
    return result.get_counts()


def run_estimator(circuit, observable, shots: int = 1024,
                  t1: float = None, t2: float = None) -> float:
    """
    Estimate expectation value <ψ|O|ψ> using shots-based simulation.
    
    Args:
        circuit: QuantumCircuit (no measurements) to prepare the state.
        observable: SparsePauliOp observable to measure.
        shots: Number of shots for estimation.
        t1: T1 relaxation time in microseconds. None = no noise.
        t2: T2 dephasing time in microseconds. None = no noise.
    
    Returns:
        Expectation value as float.
    """
    noise_model = build_noise_model(t1, t2)
    
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()
    
    estimator = EstimatorV2.from_backend(simulator)
    job = estimator.run([(circuit, observable)], precision=1.0 / np.sqrt(shots))
    result = job.result()
    return float(result[0].data.evs)


def run_circuit_display(circ: QuantumCircuit, t1: float = None, t2: float = None,
    shots: int = 1024, display: bool = True) -> dict:
    """Run a circuit, print JSON output, optionally display graphical views.
    
    Args:
        circ: The quantum circuit to run.
        t1: T1 relaxation time in microseconds.
        t2: T2 relaxation time in microseconds.
        shots: Number of shots.
        display: Whether to display circuit diagram and histogram (graphical).
    """
    counts = run_circuit(circ, t1=t1, t2=t2, shots=shots)
    
    # Always print JSON output
    results = {
        "circuit_stats": {
            "qubits": circ.num_qubits,
            "depth": circ.depth(),
            "gate_counts": dict(circ.count_ops())
        },
        "run": {
            "shots": shots,
            "counts": counts
        }
    }
    if t1 or t2:
        results["run"]["noise"] = {"t1_us": t1, "t2_us": t2}
    print(json.dumps(results, indent=2))
    
    # Only show graphical display if requested
    if display:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        circ.draw("mpl", ax=axes[0])
        plot_histogram(counts, ax=axes[1])
        plt.tight_layout()
        circ_no_meas = circ.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(circ_no_meas)
        plot_bloch_multivector(sv)
        plt.show()
    return counts


def estimate_shots(num_pixels: int, relative_error: float = 0.1, confidence: float = 0.95) -> int:
    """
    Estimate number of shots needed for image reconstruction.
    
    Using normal approximation: shots ≈ (1 / (2 * ε²)) * ln(2 / δ)
    where ε = relative_error * (1/num_pixels) and δ = 1 - confidence.
    
    For 10% relative error at 95% confidence: shots ≈ 185 * num_pixels²
    """
    delta = 1 - confidence
    epsilon = relative_error / num_pixels
    shots = int((1 / (2 * epsilon**2)) * math.log(2 / delta))
    return shots
