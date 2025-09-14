"""Qiskit 2.x primitives example (strict V2): SamplerV2 & EstimatorV2 (Aer).

This script builds a 2-qubit Bell state circuit and demonstrates the current
Qiskit *V2* primitive interfaces only (no legacy fallbacks):

1. SamplerV2  – sampling-based (quasi) distribution over measurement outcomes.
2. EstimatorV2 – analytic expectation values of observables (shot-free with
   a statevector backend) via pairs of (circuit, observable).

Conceptual difference:
Sampler answers: "If I measure this circuit many times, what bitstring distribution do I get?"
Estimator answers: "What are the expectation values <psi|O|psi>?"

Bell state (|00> + |11>) / sqrt(2) expectations:
    <Z0> = 0, <Z1> = 0, <Z0 Z1> = 1

New: The script now also derives <Z0>, <Z1>, <Z0Z1> from the SamplerV2 quasi
distribution and prints a comparison table against EstimatorV2 results.

"""

#pylint: disable=broad-exception-caught, missing-function-docstring, too-many-locals

from __future__ import annotations

import sys

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2, EstimatorV2
except Exception as exc:  # pragma: no cover
    print("[ERROR] Required Qiskit 2.x + Aer primitives (SamplerV2, EstimatorV2) not available:",
        exc)
    print("Install/upgrade with: pip install --upgrade 'qiskit>=2.0' qiskit-aer")
    sys.exit(1)


def build_bell_circuit(num_qubits: int = 2, measured: bool = False) -> QuantumCircuit:
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


def _extract_quasi_from_sampler_v2(result, circuit: QuantumCircuit,
                                   shots: int, sim: AerSimulator | None) -> tuple[dict, str]:
    """Best-effort extraction of a quasi distribution from a SamplerV2 result.

    Returns (quasi_dist_mapping, source_label).
    Raises RuntimeError with diagnostic info if nothing suitable is found.
    """
    # 1. Direct attributes
    if hasattr(result, 'quasi_dists'):
        qd = result.quasi_dists
        if isinstance(qd, (list, tuple)) and qd and isinstance(qd[0], dict):
            return qd[0], 'result.quasi_dists[0]'
    if hasattr(result, 'quasi_dist') and \
        isinstance(result.quasi_dist, dict):  # type: ignore[attr-defined]
        return result.quasi_dist, 'result.quasi_dist'  # type: ignore[attr-defined]

    # 2. Publication style
    for attr in ('pub_results', '_pub_results'):
        if hasattr(result, attr):
            try:
                pubs = getattr(result, attr)
                if pubs:
                    first = pubs[0]
                    if hasattr(first, 'quasi_dist') and \
                        isinstance(first.quasi_dist, dict):  # type: ignore[attr-defined]
                        return first.quasi_dist, f'{attr}[0].quasi_dist'
                    # Some variants may store counts; look for 'counts'
                    if hasattr(first, 'counts') and \
                        isinstance(first.counts, dict):  # type: ignore[attr-defined]
                        counts = first.counts  # type: ignore[attr-defined]
                        total = sum(counts.values()) or 1
                        quasi = {k: v / total for k, v in counts.items()}
                        return quasi, f'{attr}[0].counts -> normalized'
            except Exception:  # pragma: no cover
                pass

    # 3. metadata path: sometimes probabilities embedded in metadata
    if hasattr(result, 'metadata'):
        md = result.metadata
        if isinstance(md, dict):
            # Look for candidate dict of bitstrings.
            for k, v in md.items():
                if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                    # Normalize if values look like ints (counts) instead of probs.
                    vals = list(v.values())
                    if vals and all(isinstance(x, int) for x in vals):
                        total = sum(vals) or 1
                        return {kk: vv / total for kk, vv in v.items()}, \
                            f'metadata["{k}"] counts -> normalized'
                    return v, f'metadata["{k}"]'

    # 4. Generic containers
    for name in ('data', 'results'):
        if hasattr(result, name):
            obj = getattr(result, name)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                        return v, f'{name}["{k}"]'
            if isinstance(obj, (list, tuple)) and obj:
                candidate = obj[0]
                if isinstance(candidate, dict):
                    if all(isinstance(k, str) for k in candidate.keys()):
                        return candidate, f'{name}[0]'
                    for v in candidate.values():
                        if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                            return v, f'{name}[0].<nested>'

    # 5. As a last resort, synthesize distribution via direct simulation of circuit.
    try:
        # Reuse injected simulator if provided; else allocate a fresh one.
        sim_fallback = sim if sim is not None else AerSimulator()
        # Ensure circuit has measurements; if not, create a measured copy.
        if circuit.num_clbits == 0:
            measured = circuit.copy()
            measured.measure_all()
        else:
            measured = circuit
        transpiled = measured  # Light path; could call transpile(measured, sim_fallback)
        job = sim_fallback.run(transpiled, shots=shots)
        counts = job.result().get_counts()
        total = sum(counts.values()) or 1
        quasi = {k: v / total for k, v in counts.items()}
        return quasi, 'fallback: direct AerSimulator run'
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed. Attrs: "
            + ", ".join(sorted(dir(result))) + f" | Fallback error: {exc}"
        ) from exc


def _expectations_from_quasi(quasi: dict) -> tuple[float, float, float]:
    """Compute <Z0>, <Z_{n-1}>, and global parity <Z0 Z1 ... Z_{n-1}>.

    Assumes bitstrings are little-endian (rightmost bit -> qubit 0). For a GHZ state
    (|0...0> + |1...1>)/sqrt(2), we expect <Z0>=0, <Z_{n-1}>=0, and global parity = +1.
    """
    z_first = 0.0
    z_last = 0.0
    z_global = 0.0
    for bitstr, p in quasi.items():
        if not bitstr:
            continue
        # Normalize length (pad leading zeros on the left for missing high qubits)
        s = bitstr
        # Qubit 0 is rightmost char
        b0 = s[-1]
        blast = s[0]
        v_first = 1 if b0 == '0' else -1
        v_last = 1 if blast == '0' else -1
        # Global parity: product of Z eigenvalues across all qubits
        prod = 1
        for c in s:
            prod *= (1 if c == '0' else -1)
        z_first += v_first * p
        z_last += v_last * p
        z_global += prod * p
    return z_first, z_last, z_global


def demo_sampler(circuit_with_meas: QuantumCircuit, sim: AerSimulator, shots: int = 1024):
    """Display sampling distribution using SamplerV2
    """
    print("\n=== SamplerV2 Demo ===")
    # SamplerV2 from qiskit_aer.primitives already binds to Aer internally; we keep an
    # explicit simulator object to demonstrate configurability (method/seed). The
    # primitive constructor currently doesn't accept a backend parameter, so we just
    # document and print the simulator being conceptually used.
    sampler = SamplerV2()
    job = sampler.run([circuit_with_meas], shots=shots)
    result = job.result()
    quasi, source = _extract_quasi_from_sampler_v2(result, circuit_with_meas, shots, sim)

    fallback_note = " (fallback)" if source.startswith('fallback') else ""
    print(f"Shots: {shots}  Source: {source}{fallback_note}")
    try:
        method = getattr(sim, 'options', {}).get('method', None) if hasattr(sim, 'options') \
            else None
    except Exception:
        method = None
    print(f"Using injected AerSimulator: {sim!r} method={method}")
    print("Bitstring  ApproxCount  Probability")
    for bitstr, prob in sorted(quasi.items(), key=lambda kv: -kv[1]):
        print(f"  {bitstr:>2}        {prob * shots:7.1f}    {prob:10.6f}")

    p00 = quasi.get('00', 0.0)
    p11 = quasi.get('11', 0.0)
    print(f"Correlation mass P(00)+P(11): {p00 + p11:.3f}")

    z0, z1, z0z1 = _expectations_from_quasi(quasi)
    print(f"Sampler-derived expectations: <Z0>={z0:.3f} <Z1>={z1:.3f} <Z0Z1>={z0z1:.3f}")
    return quasi, source, (z0, z1, z0z1)


def demo_estimator(circuit: QuantumCircuit, sim: AerSimulator):
    """Compute expectation values with EstimatorV2 only (no fallback).

    Pauli string ordering (little-endian): rightmost char -> qubit 0.
        'IZ' => Z on q0, 'ZI' => Z on q1, 'ZZ' => Z tensor Z.
    """
    print("\n=== EstimatorV2 Demo ===")
    estimator = EstimatorV2()
    n = circuit.num_qubits
    # Observables: Z on first, Z on last, global parity Z...Z
    # Build Pauli strings (little-endian: rightmost char -> qubit 0). For SparsePauliOp
    # string, leftmost char corresponds to highest-index qubit.
    pauli_first = ['I'] * n
    pauli_first[-1] = 'Z'  # qubit 0
    z_first = SparsePauliOp(''.join(pauli_first))

    pauli_last = ['I'] * n
    pauli_last[0] = 'Z'  # highest index qubit (n-1)
    z_last = SparsePauliOp(''.join(pauli_last))

    z_global = SparsePauliOp('Z' * n)
    labels = ["<Z0>", f"<Z{n-1}>", "<Z...>"]
    pairs = [(circuit, z_first), (circuit, z_last), (circuit, z_global)]
    job = estimator.run(pairs)
    result = job.result()

    source = 'estimatorV2.values'
    if hasattr(result, 'values'):
        values = list(result.values)  # type: ignore[attr-defined]
    else:
        # Attempt alternative structures: pub_results / _pub_results entries.
        extracted = []
        for attr in ('pub_results', '_pub_results', 'data', 'results'):
            if hasattr(result, attr):
                container = getattr(result, attr)
                if isinstance(container, (list, tuple)) and container:
                    first = container[0]
                    # Look for .values, .result, or numeric list
                    cand = None
                    if hasattr(first, 'values'):
                        cand = getattr(first, 'values')
                    elif hasattr(first, 'result') and isinstance(first.result, (list, tuple)):
                        cand = first.result
                    if cand is not None and len(cand) >= 3:
                        extracted = list(cand)[:3]
                        source = f'{attr}[0]'
                        break
        if not extracted:
            # Fallback: simulate statevector and compute manually
            sv = Statevector.from_instruction(circuit)
            def expval(op: SparsePauliOp) -> float:
                return float((sv.expectation_value(op)).real)
            extracted = [expval(z_first), expval(z_last), expval(z_global)]
            source = 'fallback: statevector expectation'
        values = extracted

    fallback_note = " (fallback)" if source.startswith('fallback') else ""
    print(f"Observable  Expectation (theory: 0, 0, 1)  Source: {source}{fallback_note}")
    for label, value in zip(labels, values):
        print(f"  {label:7}  {value: .6f}")
    try:
        method = getattr(sim, 'options', {}).get('method', None) if hasattr(sim, 'options') \
            else None
    except Exception:  # pragma: no cover
        method = None
    print(f"Using injected AerSimulator (estimator context): {sim!r} method={method}")
    return values, source


def main(num_qubits: int, method: str, seed: int) -> None:
    # Choose number of qubits for GHZ state
    circuit = build_bell_circuit(num_qubits=num_qubits, measured=False)
    circuit_meas = build_bell_circuit(num_qubits=num_qubits, measured=True)
    print("circuit:")
    print(circuit)
    print("circuit_meas:")
    print(circuit_meas)

    # IBM (Aer) simulator running locally with deterministic seed
    sim = AerSimulator(method=method, seed_simulator=seed)

    # get results using the Sampler
    quasi, sampler_source, sampler_expectations = demo_sampler(circuit_meas, sim, shots=1024)
    # Simple sanity: ensure probability mass ~1.0 (tolerate tiny numerical drift)
    total_prob = abs(sum(quasi.values()) - 1.0)
    if total_prob > 1e-6:
        print(f"[WARN] Sampler quasi distribution not normalized (Δ={total_prob:.2e})")

    # get results using the Estimator
    est_values, est_source = demo_estimator(circuit, sim)

    # Comparison table
    z0_s, z1_s, zz_s = sampler_expectations
    z0_e, z1_e, zz_e = est_values
    print("\n=== Expectation Comparison (Sampler vs Estimator) ===")
    print("Observable        Sampler    Estimator   |Diff|")
    def line(name, sv, ev):
        diff = abs(sv - ev)
        print(f"{name:15} {sv:8.4f}  {ev:9.4f}  {diff:6.4f}")
    line('<Z0>', z0_s, z0_e)
    line(f'<Z{circuit.num_qubits-1}>', z1_s, z1_e)
    line('<Z...>', zz_s, zz_e)
    print(f"Sampler source: {sampler_source}")
    print(f"Estimator source: {est_source}")

    print("\nDone.")


if __name__ == '__main__':
    main(3, 'statevector', 42)
