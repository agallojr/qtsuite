# Uncertainty Quantification Usage Guide

## Overview

The UQ analysis requires multiple runs of the same workflow to calculate statistical metrics (mean, standard deviation, confidence intervals).

## Workflow

### 1. Run the same TOML configuration multiple times

```bash
# Run 1
python chal.py in-MeshxShots.toml

# Run 2
python chal.py in-MeshxShots.toml

# Run 3
python chal.py in-MeshxShots.toml
```

Each run creates a new workflow directory (e.g., `/tmp/lwfm/qt02-cfd/wf123`) and saves `results.pkl`.

### 2. Run UQ analysis on multiple workflow results

```bash
python uq_analysis.py /tmp/lwfm/qt02-cfd/wf123 /tmp/lwfm/qt02-cfd/wf124 /tmp/lwfm/qt02-cfd/wf125
```

This will:
- Load all case data from the specified workflows
- Group cases by parameter configuration
- Calculate mean, std, 95% CI for each configuration
- Generate a plot with confidence intervals
- Print summary statistics

## Output

The UQ analysis produces:

1. **Console output**: Summary statistics for each parameter configuration
2. **Plot**: Two subplots showing:
   - Mean fidelity with 95% confidence interval (shaded region)
   - Standard deviation across parameter sweep

## Example Output

```
UQ Summary:
--------------------------------------------------------------------------------
qc_shots=100, NQ_MATRIX=2, nx=2, ny=2
  Mean: 0.985234
  Std:  0.012456
  95% CI: [0.972778, 0.997690]
  CV:   0.0126
  N:    3

qc_shots=1000, NQ_MATRIX=2, nx=2, ny=2
  Mean: 0.998765
  Std:  0.001234
  95% CI: [0.997531, 0.999999]
  CV:   0.0012
  N:    3
```

## Notes

- Requires at least 2 runs for meaningful statistics
- More runs (5-10) give better confidence intervals
- All runs should use the same TOML configuration
- The script automatically detects the x-axis parameter from metadata
- Results are saved to `uq_analysis.png` in the parent directory of the first workflow
