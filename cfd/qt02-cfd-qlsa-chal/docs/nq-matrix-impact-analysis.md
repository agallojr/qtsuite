# NQ_MATRIX Impact on Computation Quality

## Overview

`NQ_MATRIX` is a critical parameter that controls the **precision vs. cost tradeoff** in the HHL quantum algorithm. However, **more qubits does NOT automatically mean better results**.

## What NQ_MATRIX Controls

### Direct Impact

```
Matrix Size (minimum) = 2^NQ_MATRIX
```

**Example:**
- `NQ_MATRIX = 2` → 4×4 matrix minimum
- `NQ_MATRIX = 3` → 8×8 matrix minimum  
- `NQ_MATRIX = 4` → 16×16 matrix minimum

### Actual Circuit Qubits (After Expansion)

The ORNL HHL implementation requires additional qubits for:
1. **Hermitian projection** (if A is not Hermitian)
2. **Matrix conditioning** (if κ is too large)
3. **Ancilla qubits** for QPE and controlled rotations

**Observed expansion from qt02 experiments:**

```
NQ_MATRIX = 2 → 7 actual qubits   (3.5× expansion)
NQ_MATRIX = 3 → 9 actual qubits   (3.0× expansion)
NQ_MATRIX = 4 → 11 actual qubits  (2.75× expansion)
```

## Impact on Computation Quality

### 1. **Eigenvalue Precision (QPE Phase)**

`NQ_MATRIX` determines the number of qubits used in **Quantum Phase Estimation (QPE)**, which extracts eigenvalues of the A matrix.

**Higher NQ_MATRIX:**
- ✓ More precise eigenvalue estimation
- ✓ Better resolution of closely-spaced eigenvalues
- ✓ Improved accuracy in the reciprocal calculation (1/λ)

**Trade-off:**
- ✗ Deeper circuits (more gates)
- ✗ More opportunities for gate errors
- ✗ Exponentially longer execution time

### 2. **Fidelity vs NQ_MATRIX (Empirical Results)**

From the README experiments:

> "Notice how an increase in qubits, while improving the precision of the eigenvalues in the QPE phase of the HHL algorithm, **does not automatically improve fidelity**."

**Key finding:** More qubits can actually **decrease** fidelity due to:
- Increased circuit depth → accumulated gate errors
- More qubits → higher chance of decoherence
- NISQ-era noise dominates any precision gains

### 3. **Circuit Depth Impact**

```
Circuit Depth ∝ NQ_MATRIX × (other factors)
```

**From README:**
> "Depths in the thousands are prohibitive in the NISQ era as the errors compound."

**Practical limits:**
- Clean simulators: Can handle depth > 1000
- Noisy simulators: Fidelity degrades rapidly beyond depth ~100
- Real quantum hardware (NISQ): Essentially unusable beyond depth ~50-100

### 4. **Condition Number Sensitivity**

`NQ_MATRIX` interacts with matrix conditioning:

**For hele-shaw cases:**
- Larger `NQ_MATRIX` allows representing larger, potentially better-conditioned matrices
- But if the underlying CFD problem is ill-conditioned (low `mu`, high Reynolds), no amount of qubits helps
- ORNL code may **automatically expand** matrix size if κ > threshold

**From in-ConditionNumber.toml:**
```toml
max_condition_number = 1e4
```

If κ exceeds this, the case is skipped regardless of `NQ_MATRIX`.

## Quality Metrics Affected by NQ_MATRIX

### Positive Correlations (Higher is Better)

1. **Eigenvalue Resolution**
   - More qubits → finer discretization of eigenvalue spectrum
   - Critical for matrices with closely-spaced eigenvalues

2. **Theoretical Precision**
   - QPE precision: `Δλ ≈ 2π / 2^NQ_MATRIX`
   - Doubles precision with each additional qubit

### Negative Correlations (Higher is Worse)

1. **Circuit Depth**
   ```
   Depth_transpiled ≈ O(NQ_MATRIX × log(matrix_size))
   ```
   - From scaling analysis plots, depth grows super-linearly

2. **Gate Count**
   - Total gates increases exponentially with `NQ_MATRIX`
   - Each gate introduces error in NISQ regime

3. **Execution Time**
   - Simulation time: `O(2^(total_qubits))`
   - Real hardware: Queue time + execution time increases

4. **Error Accumulation (NISQ)**
   - Gate error rate: ~0.1% - 1% per gate
   - For depth=1000, accumulated error ≈ 63% - 99.99%
   - Makes results essentially random

## Optimal NQ_MATRIX Selection Strategy

### For Clean Simulators (No Noise)

**Goal:** Balance precision vs. computational cost

```toml
# Small problems (testing)
NQ_MATRIX = 2   # Fast, good enough for 2×2 - 4×4 grids

# Medium problems
NQ_MATRIX = 3   # Sweet spot for 4×4 - 8×8 grids

# Large problems (if you have time/memory)
NQ_MATRIX = 4   # Only if matrix is well-conditioned
```

**Observed fidelity plateau:** ~0.85-0.95 regardless of `NQ_MATRIX` on clean sims

### For Noisy Simulators / Real Hardware (NISQ)

**Goal:** Minimize circuit depth to reduce error accumulation

```toml
# NISQ regime - keep it minimal
NQ_MATRIX = 2   # Maximum practical value
# NQ_MATRIX = 3   # Already too deep for useful results
```

**Observed fidelity:** ~0.1-0.3 on noisy sims, essentially random

### For Condition Number Studies

```toml
# Start small, let ORNL code expand if needed
NQ_MATRIX = 2
max_condition_number = 1e4  # Adjust based on tolerance
```

The ORNL code will automatically expand the matrix if conditioning requires it.

## Interaction with Other Parameters

### NQ_MATRIX × Shot Count

**Finding from experiments:**
- Increasing shots helps **up to a point** (diminishing returns)
- `NQ_MATRIX` affects the **ceiling** of achievable fidelity
- But higher `NQ_MATRIX` doesn't raise that ceiling in NISQ regime

**Example (from Fig5):**
```
NQ_MATRIX = 3, shots = [100, 1000, 10000, 100000, 1000000]
Fidelity: [0.65, 0.82, 0.87, 0.88, 0.88]  # Plateaus around 10k shots
```

### NQ_MATRIX × Backend Type

| Backend | Optimal NQ_MATRIX | Reason |
|---------|-------------------|--------|
| `statevector_sim_aer` | 2-4 | Deterministic, no noise, limited by memory |
| `density_matrix_sim_aer` | 2-3 | Can handle noise, but memory-intensive |
| `ibm_brisbane_aer` | 2 | Noise model makes depth critical |
| `ibm_brisbane` (real) | 2 | Real hardware errors dominate |

### NQ_MATRIX × Mesh Size (nx, ny)

**For sample-tridiag:**
- `NQ_MATRIX` directly sets matrix size
- `nx`, `ny` are largely ignored

**For hele-shaw:**
- `nx × ny` determines CFD discretization
- `NQ_MATRIX` must be large enough to represent the resulting matrix
- ORNL code may expand if mismatch

## Empirical Results Summary

### From Qubit Scaling Study (README Fig4)

**Setup:** 2×2 mesh, `NQ_MATRIX = [2, 3, 4]`, shots up to 10k

**Results:**
- All three `NQ_MATRIX` values achieved similar **maximum fidelity** (~0.85-0.90)
- `NQ_MATRIX = 2` was **fastest** (5× faster than NQ=4)
- Higher `NQ_MATRIX` showed **more variance** in results
- **Conclusion:** For this problem size, `NQ_MATRIX = 2` was optimal

### From Scaling Analysis (README Fig6)

**Observed growth rates:**

| NQ_MATRIX | Matrix Size | Circuit Qubits | Circuit Depth | Gates |
|-----------|-------------|----------------|---------------|-------|
| 2 | 4 | 7 | ~500 | ~2000 |
| 3 | 8 | 9 | ~1200 | ~6000 |
| 4 | 16 | 11 | ~2500 | ~15000 |

**Depth growth:** Approximately `O(NQ_MATRIX^2)` or worse

## Recommendations

### 1. Start Small
```toml
NQ_MATRIX = 2  # Default for most cases
```

### 2. Only Increase If:
- ✓ Matrix has closely-spaced eigenvalues (check spectrum)
- ✓ Running on clean simulator (no noise)
- ✓ Have computational resources (time/memory)
- ✓ Condition number is well-behaved (κ < 1e4)

### 3. Never Increase If:
- ✗ Running on real quantum hardware (NISQ)
- ✗ Using noisy simulators
- ✗ Matrix is ill-conditioned (high κ)
- ✗ Just trying to "make results better" (won't work)

### 4. Monitor These Metrics:
```python
# From main_workflow.py logging
logger.info(f"Circuit properties: qubits={num_qubits_circuit}, "
           f"depth={circuit_depth}, gates={circuit_size}")
logger.info(f"Matrix condition number: κ={condition_number:.4e}")
logger.info(f"Transpiled: depth={transpiled_depth}, gates={transpiled_size}")
```

If transpiled depth > 1000, consider reducing `NQ_MATRIX`.

## Conclusion

**NQ_MATRIX is NOT a "quality knob" you can turn up for better results.**

Instead, it's a **precision-cost tradeoff** where:
- Theoretical precision increases with `NQ_MATRIX`
- Practical fidelity is limited by circuit depth and noise
- In NISQ era, **smaller is often better**
- On clean simulators, there's a **sweet spot** around `NQ_MATRIX = 2-3`

The key insight from the qt02 experiments:

> "More qubits does not necessarily translate to more fidelity."

Focus instead on:
1. Well-conditioned matrices (adjust CFD parameters like `mu`)
2. Appropriate shot counts (10k-100k for convergence)
3. Clean simulators for development
4. Minimal circuit depth for NISQ hardware
