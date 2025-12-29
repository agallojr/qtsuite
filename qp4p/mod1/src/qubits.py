"""
Qubit examples
"""

import math
from typing import List

from qiskit import QuantumCircuit, QuantumRegister

from qp4p_circuit import run_circuit_display


def circuit_prep(n: int, state: bool = False, phase: float = 0.0) -> QuantumCircuit:
    """
    Make a circuit of n qubits, all in |0> or |1> state with optional phase. Measure all.
    
    Args:
        n: Number of qubits
        state: False for |0>, True for |1>
        phase: Phase angle in radians to apply (via Rz gate)
    """
    qc = QuantumCircuit(n)
    if state:
        for i in range(n):
            qc.x(i)
    if phase != 0.0:
        for i in range(n):
            qc.rz(phase, i)
    qc.measure_all()
    return qc
 

def circuit_hadamard_all(n: int) -> QuantumCircuit:
    """
    Make a circuit of n qubits, all in Hadamard state. Measure all.
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    qc.h(qr)
    qc.measure_all()
    return qc


def circuit_ghz(n: int) -> QuantumCircuit:
    """
    Make a GHZ state circuit: (|00...0> + |11...1>) / sqrt(2). Measure all.
    Hadamard on first qubit, cascading CNOTs on all others.
    """
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def circuit_amplitude_encoding(n: int, data_vector: List[float]) -> QuantumCircuit:
    """
    Prepare an amplitude encoded state.
    
    Args:
        n: Maximum number of qubits allowed
        data_vector: Normalized amplitudes (length must be power of 2, sum of squares = 1)
    
    Example:
        data_vector = [0.5, 0.5, 0.5, 0.5]  # data is normalized: (0.5)^2 * 4 = 1
    """    
    length = len(data_vector)
    if length == 0 or (length & (length - 1)) != 0:
        print(f"Error: data_vector length ({length}) must be a power of 2")
        return None
    
    qubits_needed = int(math.log2(length))
    if qubits_needed > n:
        print(f"Error: data_vector requires {qubits_needed} qubits but n={n} allows only {n}")
        return None
    
    qc = QuantumCircuit(n)
    qc.initialize(data_vector, list(range(qubits_needed)))
    qc.measure_all()
    return qc


# *****************************************************************************
# main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Qubit examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python qubits.py prep 
  python qubits.py prep -n 5 --state --phase 1.57 --t1 50 --t2 30

  python qubits.py hadamard
  python qubits.py hadamard -n 4 --t1 50 --t2 30

  python qubits.py ghz
  python qubits.py ghz -n 5 --t1 50 --t2 30

  python qubits.py amplitude --data 0.5 0.5 0.5 0.5
  python qubits.py amplitude -n 3 --data 1 0 0 0  --t1 50 --t2 30
""")
    parser.add_argument("example", choices=["prep", "hadamard", "ghz", "amplitude"],
                        help="Which example to run")
    parser.add_argument("-n", type=int, default=3,
                        help="Number of qubits (default: 3)")
    parser.add_argument("--state", action="store_true",
                        help="Prepare qubits in |1> state instead of |0>")
    parser.add_argument("--phase", type=float, default=0.0,
                        help="Phase angle in radians to apply (default: 0)")
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in µs (default: None = no noise)")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in µs (default: None = no noise)")
    parser.add_argument("--data", type=float, nargs='+', default=None,
                        help="Data vector for amplitude encoding (list of floats)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable graphical display of circuit and histogram")
    args = parser.parse_args()
    display = not args.no_display

    if args.example == "prep":
        run_circuit_display(circuit_prep(args.n, state=args.state, phase=args.phase),
            t1=args.t1, t2=args.t2, display=display)
    elif args.example == "hadamard":
        run_circuit_display(circuit_hadamard_all(args.n), t1=args.t1, t2=args.t2, display=display)
    elif args.example == "ghz":
        run_circuit_display(circuit_ghz(args.n), t1=args.t1, t2=args.t2, display=display)
    elif args.example == "amplitude":
        run_circuit_display(circuit_amplitude_encoding(args.n, args.data),
            t1=args.t1, t2=args.t2, display=display)

