# Workflow Reference

Reference information for the workflow runs used in the comparison analysis.

## Workflow Directories

| Workflow ID | Timestamp | Description |
|-------------|-----------|-------------|
| `320a98b0` | Nov 25 16:32 | Tridiagonal cases (NQ=2, NQ=5) - statevector, circuit metrics for right panel |
| `3c1b8abb` | Nov 25 17:19 | Tridiagonal NQ=4 (11 qubits) - statevector, circuit metrics for right panel |
| `ccdbece7` | Nov 25 18:50 | **Tridiagonal NQ=2,4,5 - MPS simulator with shot noise (fidelity for left panel)** |
| `f2df9b40` | Nov 25 12:07 | Hele-Shaw cases (2x2, 3x3, 4x4) with full execution and fidelity results |
| `0bbb27f2` | Nov 25 14:25 | Hele-Shaw scaling study (transpile-only, circuit metrics) |
| `919f054d` | Nov 25 20:34 | **Tridiag NQ=2 with Heron noise** - MPS simulator, shots [100, 1K, 10K, 100K] |
| `6c0cab98` | Nov 25 20:44 | **Hele-Shaw 2x2 with Heron noise** - MPS simulator, shots [100, 1K, 10K, 100K] |
| `aa531670` | Nov 25 22:02 | **Hele-Shaw 2x2 on IBM Fez (real hardware)** - shots [100, 1K, 10K, 100K] |
| `df2add62` | Nov 25 22:11 | **Hele-Shaw 2x2 on IBM Torino run1 (real hardware)** - shots [100, 1K, 10K, 100K] |
| `058cf139` | Nov 25 22:16 | **Hele-Shaw 2x2 on IBM Torino run2 (real hardware)** - shots [100, 1K, 10K, 100K] |
| `ea21306c` | Nov 25 22:21 | **Hele-Shaw 2x2 on IBM Torino run3 (real hardware)** - shots [100, 1K, 10K, 100K] |
| `8097a5fe` | Nov 25 22:47 | **Hele-Shaw 2x2 on IBM Torino run4 (real hardware)** - shots [100, 1K, 10K, 100K] |
| `4f765788` | Nov 26 14:37 | **Hele-Shaw 2x2 Heron noise sim (4 runs)** - density_matrix, generic Heron noise |
| `0e63f2d8` | Nov 26 14:45 | **Hele-Shaw 2x2 ibm_torino_aer (4 runs)** - AerSimulator.from_backend with real calibration |

## Directory Paths

```
/Users/agallojr/.lwfm/out/qt02-cfd/320a98b0/  # Tridiagonal circuit metrics (NQ=2, NQ=5)
/Users/agallojr/.lwfm/out/qt02-cfd/3c1b8abb/  # Tridiagonal circuit metrics (NQ=4)
/Users/agallojr/.lwfm/out/qt02-cfd/ccdbece7/  # Tridiagonal fidelity (MPS, shot noise)
/Users/agallojr/.lwfm/out/qt02-cfd/f2df9b40/  # Hele-Shaw fidelity runs
/Users/agallojr/.lwfm/out/qt02-cfd/0bbb27f2/  # Hele-Shaw scaling (circuit metrics)
/Users/agallojr/.lwfm/out/qt02-cfd/919f054d/  # Tridiag NQ=2 with Heron noise
/Users/agallojr/.lwfm/out/qt02-cfd/6c0cab98/  # Hele-Shaw 2x2 with Heron noise
/Users/agallojr/.lwfm/out/qt02-cfd/aa531670/  # Hele-Shaw 2x2 on IBM Fez (real)
/Users/agallojr/.lwfm/out/qt02-cfd/df2add62/  # Hele-Shaw 2x2 on IBM Torino run1 (real)
/Users/agallojr/.lwfm/out/qt02-cfd/058cf139/  # Hele-Shaw 2x2 on IBM Torino run2 (real)
/Users/agallojr/.lwfm/out/qt02-cfd/ea21306c/  # Hele-Shaw 2x2 on IBM Torino run3 (real)
/Users/agallojr/.lwfm/out/qt02-cfd/8097a5fe/  # Hele-Shaw 2x2 on IBM Torino run4 (real)
/Users/agallojr/.lwfm/out/qt02-cfd/4f765788/  # Hele-Shaw 2x2 Heron noise sim (4 runs)
/Users/agallojr/.lwfm/out/qt02-cfd/0e63f2d8/  # Hele-Shaw 2x2 ibm_torino_aer (4 runs)
```

## Data Sources for Comparison Plot

- **Fidelity data (Tridiagonal NQ=2,4,5)**: `ccdbece7/checkpoint_tridiag_nq*.json` (MPS with shot noise)
- **Fidelity data (Hele-Shaw)**: `f2df9b40/results_reconstructed.json`
- **Circuit metrics (Tridiagonal)**: `320a98b0/checkpoint_tridiag_nq*.json`, `3c1b8abb/checkpoint_*.json`
- **Circuit metrics (Hele-Shaw)**: `0bbb27f2/checkpoint_scaling_*.json`

## Cases in Each Workflow

### 320a98b0 (Tridiagonal NQ=2, NQ=5)
- `tridiag_nq2_0` through `tridiag_nq2_3`: NQ=2 with shots [100, 1000, 10000, 100000]
- `tridiag_nq5_0` through `tridiag_nq5_3`: NQ=5 with shots [100, 1000, 10000, 100000]

### 3c1b8abb (Tridiagonal NQ=4)
- `tridiag_nq4_0` through `tridiag_nq4_3`: NQ=4 (11 qubits) with shots [100, 1000, 10000, 100000]

### f2df9b40 (Hele-Shaw Fidelity)
- `hs_sv_2x2_*`, `hs_sv_3x3_*`, `hs_sv_4x4_*`: Statevector simulator runs
- `hs_brisbane_2x2_*`, `hs_brisbane_3x3_*`, `hs_brisbane_4x4_*`: Brisbane backend runs

### 0bbb27f2 (Hele-Shaw Scaling)
- `scaling_2x2`, `scaling_3x3`, `scaling_4x4`, `scaling_5x5`: Transpile-only runs for circuit metrics

## Noise Study (IBM Heron Model)

### 919f054d (Tridiag NQ=2 with Heron Noise)
- `tridiag_nq2_0` through `tridiag_nq2_3`: NQ=2 (7 qubits, 3.4K depth) with shots [100, 1K, 10K, 100K]
- Noise model: IBM Heron (0.02% 1Q depolarizing, 0.3% 2Q depolarizing, 0.5% readout)
- Results: Fidelity ~79-81%, decreasing with shots as solution converges to noisy output

### 6c0cab98 (Hele-Shaw 2x2 with Heron Noise)
- `hs_2x2_0` through `hs_2x2_3`: 2x2 grid (953 depth) with shots [100, 1K, 10K, 100K]
- Noise model: IBM Heron (same as above)
- Results: Fidelity ~99.8%, shallow circuit minimally affected by noise

### aa531670 (Hele-Shaw 2x2 on IBM Fez - Real Hardware)
- `hs_fez_2x2_0` through `hs_fez_2x2_3`: 2x2 grid (953 depth) with shots [100, 1K, 10K, 100K]
- Backend: IBM Fez (Heron-class, 156 qubits)
- Results: Fidelity ~80-88%, significantly lower than Heron sim (~99.8%)
- Real hardware shows ~12-20% more fidelity loss than noise model predicts

### df2add62 (Hele-Shaw 2x2 on IBM Torino run1 - Real Hardware)
- `hs_torino_2x2_0` through `hs_torino_2x2_3`: 2x2 grid (953 depth) with shots [100, 1K, 10K, 100K]
- Backend: IBM Torino (Heron-class, 133 qubits)
- Results: Fidelity ~80-93%, improving with shots (unusual behavior)

### 058cf139 (Hele-Shaw 2x2 on IBM Torino run2 - Real Hardware)
- `hs_torino_2x2_0` through `hs_torino_2x2_3`: 2x2 grid (953 depth) with shots [100, 1K, 10K, 100K]
- Backend: IBM Torino (Heron-class, 133 qubits)
- Results: Fidelity ~82-93%, peaks at 1K shots then drops at 100K

### ea21306c (Hele-Shaw 2x2 on IBM Torino run3 - Real Hardware)
- `hs_torino_2x2_0` through `hs_torino_2x2_3`: 2x2 grid (953 depth) with shots [100, 1K, 10K, 100K]
- Backend: IBM Torino (Heron-class, 133 qubits)
- Results: Fidelity ~78-93%, dips at 1K shots then recovers

### 8097a5fe (Hele-Shaw 2x2 on IBM Torino run4 - Real Hardware)
- `hs_torino_2x2_0` through `hs_torino_2x2_3`: 2x2 grid (953 depth) with shots [100, 1K, 10K, 100K]
- Backend: IBM Torino (Heron-class, 133 qubits)
- Results: Fidelity ~90-94%, consistently high across all shot counts
- Best performing run of the 4 Torino runs

### 4f765788 (Hele-Shaw 2x2 Heron Noise Sim - 4 runs)
- 4 independent runs with generic Heron noise model (density_matrix backend)
- Results: Fidelity ~98-99.8%, very tight CI (σ=0.006)

### 0e63f2d8 (Hele-Shaw 2x2 ibm_torino_aer - 4 runs)
- 4 independent runs using AerSimulator.from_backend(ibm_torino) with real calibration
- Results: Fidelity ~98-99%, slightly more variability than generic Heron (σ=0.009)

### Key Findings
- Circuit depth is the dominant factor in noise-induced fidelity loss
- Tridiag NQ=2 (3.4K depth): ~20% fidelity loss with Heron noise sim
- Hele-Shaw 2x2 (953 depth): <1% fidelity loss with Heron noise sim
- **Real IBM hardware shows significant variation between devices (Fez ~85%, Torino ~82-93%)**
- **Heron noise model is optimistic - real hardware shows 7-20% more fidelity loss**
- **Significant run-to-run variability on same hardware (Torino: 4 runs show different patterns)**
- Non-monotonic fidelity vs shots behavior on real hardware
- UQ analysis (n=4): Mean fidelity ~86-91%, with wide 95% CI at 1K shots

### Noise Model Comparison (UQ, n=4 runs each)
| Source | Mean Fidelity | Avg σ | Gap vs Real |
|--------|---------------|-------|-------------|
| ibm_torino_aer (real calibration) | 98.5% | 0.009 | +9.4% |
| Generic Heron noise | 99.3% | 0.006 | +10.2% |
| Real Torino Hardware | 89.1% | 0.049 | — |

**Both noise models significantly underestimate real hardware degradation (~9-10% fidelity gap)**

### Plot Output
- `noise_study_comparison.png`: Fidelity vs shots (Tridiag Heron sim, HS Heron sim, HS IBM Fez, HS IBM Torino run1/2/3)
- `uq_torino_4runs.png`: UQ analysis with 4 Torino runs showing mean ± 95% CI
- `uq_sim_vs_real.png`: Comparison of Heron noise sim vs real Torino hardware
