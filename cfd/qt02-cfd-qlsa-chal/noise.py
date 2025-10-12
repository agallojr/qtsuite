"""
Custom noise model for quantum circuit simulation.
"""

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def add_custom_noise():
    """
    Add custom noise to a sim backend.
    """
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.0000001, 1)  # 0.00001% - minimal but measurable
    noise_model.add_all_qubit_quantum_error(error_1q,
        ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])

    error_2q = depolarizing_error(0.0000005, 2)  # 0.00005% - minimal but measurable
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cy', 'cz'])

    # Keep readout errors for any sampling-based measurements
    readout_error = ReadoutError([[0.995, 0.005], [0.01, 0.99]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model
