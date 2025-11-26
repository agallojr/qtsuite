"""
Custom noise model for quantum circuit simulation.

Noise models based on IBM Quantum hardware specifications.
"""

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def add_custom_noise(model='heron'):
    """
    Add custom noise to a sim backend.
    
    Parameters
    ----------
    model : str
        Noise model to use: 'heron' (IBM Heron/Boston-class), 
        'eagle' (IBM Eagle/Brisbane-class), or 'minimal'
    
    Returns
    -------
    NoiseModel
        Qiskit Aer noise model
    """
    if model == 'heron':
        return _heron_noise_model()
    elif model == 'eagle':
        return _eagle_noise_model()
    else:
        return _minimal_noise_model()


def _heron_noise_model():
    """
    IBM Heron-class noise model (ibm_torino, ibm_fez, ibm_boston).
    
    Based on IBM Heron R2 specifications (2024):
    - Single-qubit gate error: ~0.02% (2e-4)
    - Two-qubit gate error (CZ): ~0.3% (3e-3)  
    - Readout error: ~0.5% (5e-3)
    - T1: ~300 μs, T2: ~200 μs
    - QV: 512
    
    These are significantly better than Eagle-class processors.
    """
    noise_model = NoiseModel()
    
    # Single-qubit gate errors (~0.02%)
    error_1q = depolarizing_error(0.0002, 1)
    noise_model.add_all_qubit_quantum_error(error_1q,
        ['id', 'x', 'sx', 'rz', 'h', 'ry', 'rx'])
    
    # Two-qubit gate errors (~0.3% for CZ, Heron native gate)
    error_2q = depolarizing_error(0.003, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
    
    # Readout errors (~0.5%)
    # P(0|1) = prob of reading 0 when state is 1
    # P(1|0) = prob of reading 1 when state is 0
    readout_error = ReadoutError([[0.995, 0.005], [0.005, 0.995]])
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model


def _eagle_noise_model():
    """
    IBM Eagle-class noise model (ibm_brisbane, ibm_sherbrooke).
    
    Based on IBM Eagle R3 specifications:
    - Single-qubit gate error: ~0.03% (3e-4)
    - Two-qubit gate error (ECR): ~0.8% (8e-3)
    - Readout error: ~1.5% (1.5e-2)
    - T1: ~200 μs, T2: ~150 μs
    - QV: 128
    """
    noise_model = NoiseModel()
    
    # Single-qubit gate errors (~0.03%)
    error_1q = depolarizing_error(0.0003, 1)
    noise_model.add_all_qubit_quantum_error(error_1q,
        ['id', 'x', 'sx', 'rz', 'h', 'ry', 'rx'])
    
    # Two-qubit gate errors (~0.8%)
    error_2q = depolarizing_error(0.008, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
    
    # Readout errors (~1.5%)
    readout_error = ReadoutError([[0.985, 0.015], [0.015, 0.985]])
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model


def _minimal_noise_model():
    """
    Minimal noise model for testing - nearly ideal.
    """
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.0000001, 1)
    noise_model.add_all_qubit_quantum_error(error_1q,
        ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])

    error_2q = depolarizing_error(0.0000005, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cy', 'cz'])

    readout_error = ReadoutError([[0.995, 0.005], [0.01, 0.99]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model
