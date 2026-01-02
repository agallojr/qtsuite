"""
Minimal quantum hello world example.

Creates a simple 2-qubit circuit with a Hadamard gate and CNOT - the simpliest entangled
"Bell state" - & runs it on a noiseless Aer simulator with 1024 shots, prints the measurement
results as JSON. (Later we'll show extending this to N qubits of entanglement, the "GHZ state".)

Shows the general programming pattern - make a circuit, transpile for a specific target backend,
execute, & post-process.

This will also validate that the major required libraries are installed.
"""

import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

SHOTS = 1024
OPTIMIZATION_LEVEL = 1

# Create a simple Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)           # Hadamard on qubit 0
qc.cx(0, 1)       # CNOT with control=0, target=1
qc.measure_all()  # Measure both qubits

# Set up noiseless Aer simulator - later we'll show how to simulate noise(s)
simulator = AerSimulator()

# Transpile the circuit for the target backend
transpiled_qc = transpile(qc, backend=simulator, optimization_level=OPTIMIZATION_LEVEL)

# Run the transpiled circuit with some number of shots, i.e. sample the circuit SHOTS times
# The job object contains information about the execution & counts of bitstring outcomes.
job = simulator.run(transpiled_qc, shots=SHOTS)
result = job.result() 
counts = result.get_counts()

# Print some results as JSON
output = {
    "original_circuit": {
        "qubits": qc.num_qubits,
        "depth": qc.depth(),
        "gates": dict(qc.count_ops())
    },
    "transpiled_circuit": {
        "qubits": transpiled_qc.num_qubits,
        "depth": transpiled_qc.depth(),
        "gates": dict(transpiled_qc.count_ops())
    },
    "execution": {
        "shots": SHOTS,
        "simulator": "AerSimulator (noiseless)",
        "optimization_level": OPTIMIZATION_LEVEL
    },
    "results": {
        "counts": counts
    }
}

print(json.dumps(output, indent=2))
