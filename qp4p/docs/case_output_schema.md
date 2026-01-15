# Per-Case JSON Output Schema

This document describes the standardized JSON output format for quantum algorithm case results in qp4p.

## Overview

The goal is to capture **who, what, where, when, why, and how** for each experiment:

- **Who**: `user_info` (username, hostname)
- **What**: `algorithm`, `problem`, `results`
- **Where**: `library_versions` (environment), `backend_info`
- **When**: `timestamp`
- **Why**: Elusive—human-provided context outside this schema
- **How**: `config`, `circuit_info`

If experiments, no matter where run or by who, are to be compared there needs to be some standardization. This is where the JSON output format comes in. It provides a consistent way to capture and compare experiments, or to programmatically morph between equivalent forms. The SWE AI tooling can be useful here.

(The SWE AI tooling can also be useful in porting forward codes as their underlying stack changes or is made obsolete. This may not solve all repeatability issues, but it can help.)

Each algorithm script outputs a JSON object to stdout. This enables consistent post-processing and comparison across different algorithms.

## Top-Level Structure

```json
{
  "algorithm": "string",
  "script_name": "string",
  "timestamp": "ISO 8601 datetime",
  "user_info": { ... },
  "status": "success" | "error",
  "library_versions": { ... },
  "problem": { ... },
  "config": { ... },
  "results": { ... },
  "metrics": { ... },
  "circuit_info": { ... },
  "backend_info": { ... },
  "error": "string (optional)"
}
```

## Field Descriptions

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | string | Algorithm identifier (e.g., `"vqls"`, `"hhl"`, `"hhl_qrisp"`, `"cks"`) |
| `script_name` | string | Source script filename (e.g., `"ax_equals_b_vqls.py"`) |
| `timestamp` | string | ISO 8601 UTC timestamp (e.g., `"2026-01-15T18:44:54.821836Z"`) |
| `status` | string | Execution status: `"success"` or `"error"` |

### `user_info`

```json
{
  "username": "string",
  "hostname": "string"
}
```

### `library_versions`

Dictionary mapping package names to version strings. Includes all installed packages plus Python version. Sorted alphabetically.

```json
{
  "numpy": "2.3.5",
  "qiskit": "2.3.0",
  "python": "3.12.10",
  ...
}
```

---

## Problem Definition (`problem`)

Describes the input problem. **Structure is algorithm-specific.** Example for linear system solvers:

```json
{
  "matrix": [[2.27, 0.11], [0.11, 4.74]],
  "rhs": [-0.234, -0.234],
  "dimension": 2,
  "condition_number": 2.098
}
```

Algorithms may add fields (e.g., `eigenvalues`, `scale_factor`).

---

## Configuration (`config`)

**Algorithm-specific.** Contains parameters passed to the script. Example:

```json
{
  "shots": 1024,
  "precision": 3,
  "backend": "aer_simulator"
}
```

---

## Results (`results`)

Contains both classical and quantum solutions in multiple formats for comparison.

### Standard Fields (All Linear Solvers)

| Field | Type | Description |
|-------|------|-------------|
| `classical_solution` | array[float] | Classical solution x = A⁻¹b (raw, unnormalized) |
| `classical_solution_normalized` | array[float] | Classical solution normalized to unit length |
| `quantum_solution` | array[float] | Quantum solution (raw amplitudes or scaled) |
| `quantum_solution_normalized` | array[float] | Quantum solution normalized to unit length |

**Notes:**
- `classical_solution` is the exact solution from `np.linalg.solve(A, b)`
- `quantum_solution` format varies by algorithm:
  - **VQLS**: Scaled to approximate classical magnitude
  - **HHL/HHL-Qrisp/CKS**: Raw amplitudes from quantum state
- Normalized versions enable direct comparison of solution directions

---

## Metrics (`metrics`)

**Algorithm-specific.** Computed by each script during post-processing. Example:

```json
{
  "fidelity": 0.9999,
  "l2_error": 0.0001,
  "optimization_iterations": 53
}
```

Common metrics include `fidelity` and `l2_error`; algorithms may add others.

---

## Circuit Information (`circuit_info`)

**Algorithm-specific.** Quantum circuit statistics. Example:

```json
{
  "num_qubits": 5,
  "depth": 47,
  "gate_counts": { "cx": 22, "h": 8, "ry": 4 }
}
```

Algorithms may add fields (e.g., `num_parameters`, `transpiled_stats`).

---

## Backend Information (`backend_info`)

Optional. Captures execution environment details. Example:

```json
{
  "name": "fake_jakarta",
  "num_qubits": 7,
  "basis_gates": ["cx", "id", "rz", "sx", "x"],
  "coupling_map": [[0,1], [1,0], [1,2], ...],
  "qubit_properties": {
    "0": { "t1_us": 150.32, "t2_us": 85.21 },
    "1": { "t1_us": 142.18, "t2_us": 78.44 }
  },
  "gate_errors": {
    "cx": [
      { "qubits": [0, 1], "error": 0.008234 },
      { "qubits": [1, 2], "error": 0.009102 }
    ],
    "sx": [
      { "qubits": [0], "error": 0.000312 }
    ]
  }
}
```

Fields vary by experiment—may include per-qubit coherence times, per-gate error rates by qubit pair, coupling topology, etc.

---

## Error Handling

When `status` is `"error"`, the `error` field contains a description:

```json
{
  "status": "error",
  "error": "Matrix is not Hermitian"
}
```

---

## Example: Complete VQLS Output

```json
{
  "algorithm": "vqls",
  "script_name": "ax_equals_b_vqls.py",
  "timestamp": "2026-01-15T18:44:54.821836Z",
  "user_info": {
    "username": "agallojr",
    "hostname": "MH-DT9TLJQR2V"
  },
  "status": "success",
  "library_versions": { "qiskit": "2.3.0", "numpy": "2.3.5", ... },
  "problem": {
    "matrix": [[2.27, 0.11], [0.11, 4.74]],
    "rhs": [-0.234, -0.234],
    "dimension": 2,
    "condition_number": 2.098
  },
  "config": {
    "ansatz_reps": 3,
    "maxiter": 200,
    "t1": null,
    "t2": null,
    "backend": null
  },
  "results": {
    "classical_solution": [-0.101, -0.047],
    "classical_solution_normalized": [-0.906, -0.422],
    "quantum_solution": [0.101, 0.047],
    "quantum_solution_normalized": [0.906, 0.422]
  },
  "metrics": {
    "fidelity": 0.9999,
    "l2_error": 0.0001,
    "optimization_iterations": 53,
    "optimization_cost": 0.0,
    "optimization_success": true
  },
  "circuit_info": {
    "num_qubits": 1,
    "depth": 4,
    "num_parameters": 4,
    "transpiled_stats": {
      "depth": 4,
      "gate_counts": { "rz": 2, "sx": 2 },
      "num_qubits": 1
    }
  }
}
```
