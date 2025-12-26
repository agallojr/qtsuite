"""
Phase Kickback Example

Demonstrates phase kickback: when a controlled gate acts on an eigenstate of the
target gate, the phase is "kicked back" to the control qubit.

Example: Apply CZ (controlled-Z) where the target is in |1⟩ state.
The Z gate has eigenvalue -1 for |1⟩, so the phase -1 kicks back to the control.
"""

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from qp4p_circuit import run_circuit


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


def run_example_kickback(t1: float = None, t2: float = None):
    """Run both circuits and compare results."""
    # With kickback (X gate on target)
    qc_kick = make_circuit_phase_kickback(x_gate=True)
    qc_kick.measure_all()
    
    # Without kickback (no X gate)
    qc_no_kick = make_circuit_phase_kickback(x_gate=False)
    qc_no_kick.measure_all()
    
    # Run simulations using helper
    counts_kick = run_circuit(qc_kick, t1=t1, t2=t2)
    counts_no_kick = run_circuit(qc_no_kick, t1=t1, t2=t2)
    
    print("With phase kickback (target in |1⟩):")
    print(f"  Counts: {counts_kick}")
    print("  Control qubit flips to |1⟩ due to kickback\n")
    
    print("Without phase kickback (target in |0⟩):")
    print(f"  Counts: {counts_no_kick}")
    print("  Control qubit stays in |0⟩\n")
    
    # Plot
    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    qc_kick_draw = make_circuit_phase_kickback(x_gate=True)
    qc_kick_draw.draw("mpl", ax=axes[0, 0])
    axes[0, 0].set_title("With Kickback (target=|1⟩)")
    
    plot_histogram(counts_kick, ax=axes[0, 1])
    axes[0, 1].set_title("Results: control -> |1⟩")
    
    qc_no_kick_draw = make_circuit_phase_kickback(x_gate=False)
    qc_no_kick_draw.draw("mpl", ax=axes[1, 0])
    axes[1, 0].set_title("No Kickback (target=|0⟩)")
    
    plot_histogram(counts_no_kick, ax=axes[1, 1])
    axes[1, 1].set_title("Results: control -> |0⟩")
    
    plt.tight_layout()
    plt.show()


# *****************************************************************************
# main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase kickback example")
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in µs (default: None = no noise)")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in µs (default: None = no noise)")
    args = parser.parse_args()
    run_example_kickback(t1=args.t1, t2=args.t2)
