"""
Minimal quantum hello world example.

Creates a simple 2-qubit circuit with a Hadamard gate and CNOT - the simpliest entangled
"Bell state" - & runs it on a noiseless Aer simulator with 1024 shots, prints the measurement
results as JSON. (Later we'll show extending this to N qubits of entanglement, the "GHZ state".)

Shows the general programming pattern - make a circuit, transpile for a specific target backend,
execute, & post-process.

This will also validate that the major required libraries are installed.
"""

import argparse
from qiskit import QuantumCircuit
from qp4p_output import output_json
from qp4p_args import add_standard_quantum_args
from qp4p_circuit import run_circuit_display


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bell state example")
    add_standard_quantum_args(parser)
    args = parser.parse_args()
    
    # Create a simple Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)           # Hadamard on qubit 0
    qc.cx(0, 1)       # CNOT with control=0, target=1
    qc.measure_all()  # Measure both qubits
    
    # Run circuit and get transpile/execution results
    result = run_circuit_display(qc, args=args, display=False)
    
    # Create standardized output and print to stdout
    output_json(
        algorithm="bell_state",
        problem={
            "description": "Simple 2-qubit Bell state (maximally entangled state)"
        },
        config_args=args,
        original_circuit=qc,
        transpile_result=result["transpile_result"],
        results_data=result["results_data"]
    )

