# Circuit Analysis: Tridiagonal vs Hele-Shaw

Comparison of HHL circuit parameters and complexity metrics.

## Summary Table

| Case | Matrix Size | Qubits | Transpiled Depth | Transpiled Gates | Condition Number (κ) |
|------|-------------|--------|------------------|------------------|---------------------|
| Tridiag NQ=2 | 4 | 7 | 3,417 | 4,643 | 3.34 |
| Hele-Shaw 2x2 | 4 | 7 | 953 | 1,263 | 1.0 |
| Tridiag NQ=5 | 32 | 13 | 2,970,897 | 4,104,003 | 4.95 |
| Hele-Shaw 3x3 | 32 (padded from 9) | 13 | 2,969,528 | 4,102,279 | 43.1 |
| Hele-Shaw 4x4 | 32 (padded from 16) | 15 | 11,887,051 | 16,414,610 | 160.2 |

## Tridiagonal Cases

### NQ=2 (4x4 matrix)
```
Matrix size:          4
Circuit qubits:       7
Transpiled depth:     3,417
Transpiled gates:     4,643
Condition number:     3.34
Circuit gen time:     5.6s
Circuit build time:   0.03s
Execution time:       1.1s
```

### NQ=5 (32x32 matrix)
```
Matrix size:          32
Circuit qubits:       13
Transpiled depth:     2,970,897
Transpiled gates:     4,104,003
Condition number:     4.95
Circuit gen time:     54.4s
Circuit build time:   35.4s
Execution time:       100.7s
```

## Hele-Shaw Cases

### 2x2 Grid (4x4 matrix)
```
Original matrix size: 4
Padded matrix size:   4
Circuit qubits:       7
Transpiled depth:     953
Transpiled gates:     1,263
Condition number:     1.0
Circuit gen time:     10.6s
Circuit build time:   0.03s
```

### 3x3 Grid (9 -> 32 padded)
```
Original matrix size: 9
Padded matrix size:   32
Circuit qubits:       13
Transpiled depth:     2,969,528
Transpiled gates:     4,102,279
Condition number:     43.1
Circuit gen time:     53.5s
Circuit build time:   34.7s
```

### 4x4 Grid (16 -> 32 padded)
```
Original matrix size: 16
Padded matrix size:   32
Circuit qubits:       15
Transpiled depth:     11,887,051
Transpiled gates:     16,414,610
Condition number:     160.2
Circuit gen time:     214.8s
Circuit build time:   144.2s
```

## Key Observations

### Circuit Complexity Scaling
- **Tridiag NQ=2 vs Hele-Shaw 2x2**: Same matrix size (4), same qubits (7), but Tridiag has ~3.6x deeper circuit
- **Tridiag NQ=5 vs Hele-Shaw 3x3**: Same padded matrix size (32), same qubits (13), nearly identical depth (~3M)
- **Hele-Shaw 4x4**: Despite same padded size as 3x3, has 4x deeper circuit (12M vs 3M)

### Condition Number Impact
- Tridiagonal matrices are well-conditioned (κ < 5) regardless of size
- Hele-Shaw matrices become increasingly ill-conditioned with grid size:
  - 2x2: κ = 1.0 (trivial)
  - 3x3: κ = 43.1
  - 4x4: κ = 160.2

### Fidelity Correlation
- Low κ (Tridiag, Hele-Shaw 2x2): High fidelity achievable with fewer shots
- High κ (Hele-Shaw 3x3, 4x4): Fidelity degrades significantly, more shots needed

## Input Parameters

### Tridiagonal
```yaml
case: sample-tridiag
NQ_MATRIX: [2, 5]  # determines matrix size as 2^NQ_MATRIX
```

### Hele-Shaw
```yaml
case: hele-shaw
nx: [2, 3, 4]  # grid points in x
ny: [2, 3, 4]  # grid points in y
D: 1           # domain depth
L: 1           # domain length
P_in: 200      # inlet pressure
P_out: 0       # outlet pressure
mu: 1.0        # viscosity
rho: 1         # density
var: pressure  # solution variable
```
