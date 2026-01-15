"""
Phase Kickback Example

Demonstrates phase kickback: when a controlled gate acts on an eigenstate of the
target gate, the phase is "kicked back" to the control qubit.

Example: Apply CZ (controlled-Z) where the target is in |1⟩ state.
The Z gate has eigenvalue -1 for |1⟩, so the phase -1 kicks back to the control.

Sample execution:
    python src/phase_kickback.py
    python src/phase_kickback.py --t1 50 --t2 40
"""

import argparse
import json
from qiskit import QuantumCircuit
from qp4p_circuit import run_circuit
from qp4p_output import create_standardized_output, output_json
from qp4p_args import add_standard_quantum_args


# *****************************************************************************

def make_circuit_phase_kickback(x_gate: bool) -> QuantumCircuit:
    """
    Demonstrate phase kickback using CZ gate.
    
    Setup:
    - Control qubit (q0): Put in superposition with H
    - Target qubit (q1): Put in |1⟩ state with X - optionally, to show with and without
    - Apply CZ gate
    
    Result: If x was applied setting the state, the -1 phase from Z|1⟩ = -|1⟩ kicks back
    to the control qubit, flipping its phase: |+⟩ -> |-⟩
    """
    qc = QuantumCircuit(2)
    # Put control in superposition: |0⟩ -> |+⟩ = (|0⟩ + |1⟩)/√2
    qc.h(0)
    if x_gate:
        # Put target in |1⟩ eigenstate of Z
        qc.x(1)
    # Apply CZ - phase kicks back to control
    qc.cz(0, 1)
    # The control is now in |-⟩ = (|0⟩ - |1⟩)/√2
    # Apply H to convert back to computational basis to observe
    qc.h(0)
    return qc


def run_example_kickback(t1: float = None, t2: float = None, backend: str = None):
    """Run both circuits and compare results."""
    # With kickback (X gate on target)
    qc_kick = make_circuit_phase_kickback(x_gate=True)
    qc_kick.measure_all()
    
    # Without kickback (no X gate)
    qc_no_kick = make_circuit_phase_kickback(x_gate=False)
    qc_no_kick.measure_all()
    
    # Run simulations using helper
    result_kick = run_circuit(qc_kick, t1=t1, t2=t2, backend=backend, coupling_map="default")
    result_no_kick = run_circuit(qc_no_kick, t1=t1, t2=t2, backend=backend, coupling_map="default")
    
    # Build results with visualization data
    output = create_standardized_output(
        algorithm="phase_kickback",
        script_name="phase_kickback.py",
        problem={
            "description": "Demonstrate phase kickback with CNOT gate"
        },
        config={
            "t1": t1,
            "t2": t2,
            "backend": backend
        },
        results={
            "with_kickback": {
                "counts": result_kick["counts"],
                "transpiled_stats": result_kick["transpiled_stats"],
                "description": "Target in |1⟩ - control flips to |1⟩"
            },
            "without_kickback": {
                "counts": result_no_kick["counts"],
                "transpiled_stats": result_no_kick["transpiled_stats"],
                "description": "Target in |0⟩ - control stays in |0⟩"
            }
        },
        backend_info=json.dumps(result_kick["backend_info"], separators=(',', ':')) if result_kick["backend_info"] else None
    )
    
    output_json(output)
    
    return output


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase kickback example")
    add_standard_quantum_args(parser, include_shots=False)
    args = parser.parse_args()
    run_example_kickback(t1=args.t1, t2=args.t2, backend=args.backend)
