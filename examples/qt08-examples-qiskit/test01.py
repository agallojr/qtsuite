"""GHZ ("n‑qubit Bell") example using explicit AerSimulator execution only.

What this script shows (Qiskit 2.x + Aer):
1. Build an n‑qubit GHZ state circuit (optionally with measurements for sampling).
2. Obtain a sampling ("quasi") distribution by running the measured circuit on an
    `AerSimulator` (shots based) and normalising counts.
3. Compute expectation values from the quasi distribution. 
   An expectation value <Z_i> is computed as:

   <Z_i> = sum_{x} ( (-1)^{x_i} * P(x) )

   where P(x) is the probability of measuring bitstring x, and x_i is the value of the
   i-th bit in x.

4. Compute analytic expectation values of Pauli observables directly from the
    ideal statevector constructed with `Statevector.from_instruction` (i.e. *without*
    using EstimatorV2). This mirrors what an Estimator primitive would return on a
    noiseless statevector backend while keeping full control in user code.

Observables reported:
    <Z0>, <Z_{n-1}>, pairwise <Z0Z1>, and the full parity <Z⊗n> (written as <Z...>).

For an ideal GHZ (|0…0> + |1…1>)/√2 we expect for n ≥ 2:
    <Zk> = 0 for any single qubit k
    <ZiZj> = +1 for any pair i,j
    Parity <Z⊗n> = 1 if n is even else 0 (because eigenvalue on |1…1> is (-1)^n)

Note on design: Earlier iterations used SamplerV2 / EstimatorV2 primitives. They have
been removed for clarity here; the example intentionally demonstrates the *equivalent*
capabilities with direct backend control plus `Statevector` utilities.
"""

#pylint: disable=broad-exception-caught, missing-function-docstring, too-many-locals

from __future__ import annotations

import sys

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import SaveStatevector
    # EstimatorV2 removed in this variant; we compute expectations directly via statevector
except Exception as exc:  # pragma: no cover
    print("[ERROR] Required Qiskit 2.x + Aer primitives (SamplerV2, EstimatorV2) not available:",
        exc)
    print("Install/upgrade with: pip install --upgrade 'qiskit>=2.0' qiskit-aer")
    sys.exit(1)


def build_bell_circuit(num_qubits: int, measured: bool = False) -> QuantumCircuit:
    """Return an n-qubit GHZ ("n-qubit Bell") state circuit.

    State prepared (ideal): (|0...0> + |1...1>) / sqrt(2)

    Parameters
    ----------
    num_qubits: Number of qubits (>=2) to entangle.
    measured:   If True, add measurements for all qubits into corresponding classical bits.
                Required for sampling distributions.
    """
    if num_qubits < 2:
        raise ValueError("num_qubits must be >= 2 for a GHZ/Bell generalization")
    qc = QuantumCircuit(num_qubits, num_qubits if measured else 0, name=f"GHZ_{num_qubits}")
    # Create GHZ: H on qubit 0, then CNOT(0 -> i) for each remaining qubit.
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    if measured:
        for i in range(num_qubits):
            qc.measure(i, i)
    return qc


def _extract_quasi_from_samples(result) -> dict:
    """
    Returns (quasi_dist_mapping, source_label).
    """
    counts = result.get_counts()          # e.g. {'000': 517, '111': 507}
    total = sum(counts.values()) or 1
    quasi = {bitstr: c / total for bitstr, c in counts.items()}
    return quasi


def _expectations_from_quasi(quasi: dict) -> tuple[float, float, float, float]:
    """Compute <Z0>, <Z_{n-1}>, pairwise <Z0Z1>, and global parity <Z0 Z1 ... Z_{n-1}>.

    Bitstrings are interpreted little-endian (rightmost bit -> qubit 0).
    """
    z_first = 0.0
    z_last = 0.0
    z_pair = 0.0
    z_global = 0.0
    for bitstr, p in quasi.items():
        if not bitstr:
            continue
        s = bitstr
        # Single-qubit Z eigenvalues for first (qubit 0) and last (qubit n-1)
        b0 = s[-1]
        blast = s[0]
        v_first = 1 if b0 == '0' else -1
        v_last = 1 if blast == '0' else -1
        # Pairwise Z0Z1 if at least two qubits present
        if len(s) >= 2:
            b1 = s[-2]
            v1 = 1 if b1 == '0' else -1
            z_pair += (v_first * v1) * p
        # Global parity
        prod = 1
        for c in s:
            prod *= (1 if c == '0' else -1)
        z_first += v_first * p
        z_last += v_last * p
        z_global += prod * p
    return z_first, z_last, z_pair, z_global


def demo_sampler(tcirc: QuantumCircuit, sim: AerSimulator, shots: int = 1024):
    """
    measurement-based
    we could have used SamplerV2 for this task which is wired into the Aer simulator,
    but we'll uses an explicit reference to the sim which comes pre-configured.
    """
    # transpile for this configured simulator
    job = sim.run(tcirc, shots=shots)
    result = job.result()
    quasi = _extract_quasi_from_samples(result)

    print("Bitstring  ApproxCount  Probability")
    for bitstr, prob in sorted(quasi.items(), key=lambda kv: -kv[1]):
        print(f"  {bitstr:>2}        {prob * shots:7.1f}    {prob:10.6f}")

    total_mass = sum(quasi.values())
    print(f"Total probability: {total_mass:.6f}")

    z0, z_last, z_pair, z_parity = _expectations_from_quasi(quasi)
    print(
        f"Sampler-derived expectations: <Z0>={z0:.3f} <Z{tcirc.num_qubits-1}>={z_last:.3f} "
        f"<Z0Z1>={z_pair:.3f} <Z...>={z_parity:.3f}\n"
    )
    return quasi, (z0, z_last, z_pair, z_parity)


def demo_estimator(tcirc: QuantumCircuit, sim: AerSimulator):
    """Compute expectation values by executing the circuit on the provided simulator.

    We explicitly run the (unmeasured) circuit on the `AerSimulator` in statevector
    mode. To guarantee the statevector is present in the result payload we append a
    `save_statevector` instruction before execution. This keeps the function structure
    close to what an Estimator-backed flow would look like while still offering full
    backend control (e.g. future insertion of noise models or dynamic circuits).
    """
    # Append an explicit save instruction so Result definitely includes the statevector.
    tcirc.append(SaveStatevector(num_qubits=tcirc.num_qubits), list(range(tcirc.num_qubits)))
    job = sim.run(tcirc)
    res = job.result()
    # Retrieve the saved statevector (single experiment => index 0)
    sv = Statevector(res.get_statevector(0))

    # Build Pauli operators
    pauli_first = ['I'] * tcirc.num_qubits
    pauli_first[-1] = 'Z'
    z_first = SparsePauliOp(''.join(pauli_first))

    pauli_last = ['I'] * tcirc.num_qubits
    pauli_last[0] = 'Z'
    z_last = SparsePauliOp(''.join(pauli_last))

    z_global = SparsePauliOp('Z' * tcirc.num_qubits)
    # Pairwise Z0Z1 operator (if >=2 qubits)
    if tcirc.num_qubits >= 2:
        pauli_pair = ['I'] * tcirc.num_qubits
        pauli_pair[-1] = 'Z'  # qubit 0
        pauli_pair[-2] = 'Z'  # qubit 1
        z_pair = SparsePauliOp(''.join(pauli_pair))
    else:  # pragma: no cover - degenerate case not expected here
        z_pair = None

    labels = ["<Z0>", f"<Z{tcirc.num_qubits-1}>", "<Z0Z1>", "<Z...>"]

    def expval(op: SparsePauliOp) -> float:
        return float((sv.expectation_value(op)).real)

    val_z0 = expval(z_first)
    val_zlast = expval(z_last)
    val_pair = expval(z_pair) if z_pair is not None else 0.0
    val_parity = expval(z_global)
    values = (val_z0, val_zlast, val_pair, val_parity)
    print("Estimator:")
    for label, value in zip(labels, values):
        print(f"  {label:7}  {value: .6f}")
    return values


def main(num_qubits: int, method: str, seed: int) -> None:
    # Choose number of qubits for GHZ state
    circuit = build_bell_circuit(num_qubits=num_qubits, measured=False)
    circuit_meas = build_bell_circuit(num_qubits=num_qubits, measured=True)
    print("circuit (w measurements):")
    print(circuit_meas)

    # IBM (Aer) simulator running locally with deterministic seed
    sim = AerSimulator(method=method, seed_simulator=seed)
    t_circuit = transpile(circuit, sim, optimization_level=0)
    t_circuit_meas = transpile(circuit_meas, sim, optimization_level=0)

    # get results using the Sampler
    quasi, sampler_expectations = demo_sampler(t_circuit_meas, sim, shots=1024)
    # Simple sanity check: ensure probability mass ~1.0 (tolerate tiny numerical drift)
    total_prob = abs(sum(quasi.values()) - 1.0)
    if total_prob > 1e-6:
        print(f"[WARN] Sampler quasi distribution not normalized (Δ={total_prob:.2e})")

    # get results using the Estimator
    est_values = demo_estimator(t_circuit, sim)

    # Comparison table
    z0_s, z_last_s, zpair_s, zpar_s = sampler_expectations
    z0_e, z_last_e, zpair_e, zpar_e = est_values
    print("\n=== Expectation Comparison (Sampler vs Estimator) ===")
    print("Observable        Sampler    Estimator   |Diff|")
    def line(name, sv, ev):
        diff = abs(sv - ev)
        print(f"{name:15} {sv:8.4f}  {ev:9.4f}      {diff:6.4f}")
    line('<Z0>', z0_s, z0_e)
    line(f'<Z{circuit.num_qubits-1}>', z_last_s, z_last_e)
    line('<Z0Z1>', zpair_s, zpair_e)
    line('<Z...>', zpar_s, zpar_e)

    print("\n")


if __name__ == '__main__':
    main(6, 'statevector', 42)
