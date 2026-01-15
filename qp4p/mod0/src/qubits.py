"""
Qubit examples.
To see the arguments, run:
python mod0/src/qubits.py -h

There are examples of quantum state preparation, use of Hadamard gates for superposition,
a GHZ state to demonstrate entanglement, and more. You can pass in arguments to set error, 
or run on an idealized simulator.
"""

import argparse
import math
from typing import List
from qiskit import QuantumCircuit, QuantumRegister
from qp4p_circuit import run_circuit_display
from qp4p_args import add_standard_quantum_args
from qp4p_output import output_json


def circiuit_state_prep(n: int, state: bool = False) -> QuantumCircuit:
    """
    Make a circuit of n qubits, all in |0> or |1> state. Measure all.
    
    Args:
        n: Number of qubits
        state: False for |0>, True for |1>
    """
    qc = QuantumCircuit(n)
    if state:
        for i in range(n):
            qc.x(i)
    qc.measure_all()
    return qc
 

def circuit_hadamard_all(n: int) -> QuantumCircuit:
    """
    Make a circuit of n qubits, all in Hadamard state - full superposition. Measure all.
    On measure each qubit has equal & independent probability of being 0 or 1.
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
    The qubits are now entangled - measuring one determines all others.
    Note: When visualizing individual Bloch spheres, each qubit appears as a dot at the
    center (maximally mixed reduced state), which is characteristic of maximal entanglement.
    The global state is pure, but each qubit individually has no definite state.
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
    
    Amplitude encoding maps classical data into quantum amplitudes, allowing exponential
    compression: 2^n classical values fit into n qubits. Each value becomes the probability
    amplitude of a basis state. When measured, the quantum state collapses to one basis state
    with probability equal to the square of its amplitude.
    
    Args:
        n: Maximum number of qubits allowed
        data_vector: Normalized amplitudes (length must be power of 2, sum of squares = 1)
    
    Example:
        data_vector = [0.5, 0.5, 0.5, 0.5]  # 4 values → 2 qubits (2^2 = 4)
        # Creates state: 0.5|00> + 0.5|01> + 0.5|10> + 0.5|11>
        # Normalized: (0.5)^2 * 4 = 1.0
        # Each basis state has 25% measurement probability
        
        data_vector = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]  # 8 values → 3 qubits (2^3 = 8)
        # Creates state: 0.5|000> + 0.5|010> + 0.5|100> + 0.5|110>
        # Only even-indexed states have non-zero amplitude
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
    parser = argparse.ArgumentParser(
        description="Qubit examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python mod0/src/qubits.py --example prep 
  python mod0/src/qubits.py --example prep -n 5 --state 1 --t1 50 --t2 30

  python mod0/src/qubits.py --example hadamard
  python mod0/src/qubits.py --example hadamard -n 4 --t1 50 --t2 30

  python mod0/src/qubits.py --example ghz
  python mod0/src/qubits.py --example ghz -n 5 --t1 50 --t2 30

  python mod0/src/qubits.py --example amplitude --data 0.5 0.5 0.5 0.5
  python mod0/src/qubits.py --example amplitude -n 3 --data 1 0 0 0  --t1 50 --t2 30
""")
    parser.add_argument("--example", choices=["prep", "hadamard", "ghz", "amplitude"],
                        required=True, help="Which example to run")
    parser.add_argument("-n", "--n", type=int, default=3,
                        help="Number of qubits (default: 3)")
    parser.add_argument("--state", type=int, choices=[0, 1], default=0,
                        help="Prepare qubits in |0> or |1> state (default: 0)")
    add_standard_quantum_args(parser, default_shots=1024)
    parser.add_argument("--data", type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5],
                        help="Data vector for amplitude encoding (default: [0.5, 0.5, 0.5, 0.5])")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable graphical display of circuit and histogram")
    args = parser.parse_args()
    display = not args.no_display

    if args.example == "prep":
        qc = circiuit_state_prep(args.n, state=args.state)
        algorithm = f"state_prep_{args.state}"
        problem = {"description": f"Prepare {args.n} qubits in |{args.state}> state", "n": args.n, "state": args.state}
    elif args.example == "hadamard":
        qc = circuit_hadamard_all(args.n)
        algorithm = "hadamard_superposition"
        problem = {"description": f"Hadamard superposition on {args.n} qubits", "n": args.n}
    elif args.example == "ghz":
        qc = circuit_ghz(args.n)
        algorithm = "ghz_state"
        problem = {"description": f"GHZ entangled state with {args.n} qubits", "n": args.n}
    else:  # amplitude
        qc = circuit_amplitude_encoding(args.n, args.data)
        algorithm = "amplitude_encoding"
        problem = {"description": f"Amplitude encoding with {len(args.data)} values", "n": args.n, "data": args.data}
    
    # Run circuit and get transpile/execution results
    result = run_circuit_display(qc, args=args, display=display)
    
    # Output standardized JSON
    output_json(
        algorithm=algorithm,
        problem=problem,
        config_args=args,
        original_circuit=qc,
        transpile_result=result["transpile_result"],
        results_data=result["results_data"]
    )
