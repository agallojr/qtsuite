# Parameter Mutual Exclusivity: `sample-tridiag` vs `hele-shaw`

## Case Type Overview

The `case` parameter determines which problem type the ORNL circuit_HHL.py code solves, and this creates **mutual exclusivity** in how other parameters are interpreted.

## Parameter Comparison

### Common Parameters (Used by Both)

Both case types use these parameters from the TOML/YAML:

```toml
NQ_MATRIX = 2          # Number of qubits for matrix (size = 2^NQ_MATRIX)
nx = 2                 # Grid points in x-direction  
ny = 2                 # Grid points in y-direction
var = "pressure"       # Variable to solve for
```

### `sample-tridiag` Specific Behavior

**For `sample-tridiag`:**
- **Simplified test case** - creates a basic tridiagonal matrix
- **CFD parameters are IGNORED**: `P_in`, `P_out`, `U_top`, `U_bottom`, `L`, `D`, `mu`, `rho`
- Only `NQ_MATRIX` directly determines matrix size
- `nx`, `ny` may be used but don't represent actual CFD mesh
- **Purpose**: Testing HHL algorithm without CFD complexity

### `hele-shaw` Specific Behavior

**For `hele-shaw`:**
- **Full CFD problem** - discretizes Hele-Shaw flow equations
- **ALL CFD parameters are USED**:
  - `P_in`, `P_out` - Boundary pressure conditions
  - `U_top`, `U_bottom` - Boundary velocity conditions
  - `L` - Physical length of channel
  - `D` - Physical width/depth of channel
  - `mu` - Dynamic viscosity (affects condition number!)
  - `rho` - Fluid density
  - `nx × ny` - Actual CFD mesh resolution
- Matrix size is **derived from** discretized PDEs, not just `NQ_MATRIX`
- `NQ_MATRIX` may be expanded if conditioning requires it

## Key Mutual Exclusivity Patterns

### 1. **CFD Physics Parameters**

**Mutually exclusive usage:**
- `sample-tridiag`: These are **template placeholders** (ignored by ORNL code)
- `hele-shaw`: These **directly affect** the A matrix and condition number

### 2. **Viscosity (`mu`) Impact**

**For `hele-shaw` only:**
- Lower `mu` → higher Reynolds number → advection-dominated flow
- Creates **ill-conditioned matrices** (high κ)
- May trigger automatic matrix expansion
- **Mutually exclusive** with `sample-tridiag` where `mu` is ignored

### 3. **Physical Domain Size (`L`, `D`)**

**Mutually exclusive:**
- `hele-shaw`: Varying `L` and `D` changes physical problem, affects matrix
- `sample-tridiag`: `L` and `D` are **not used** at all

### 4. **YAML Document Structure**

**Mutually exclusive file format:**
- `hele-shaw`: Requires **two-document YAML** (ORNL code reads doc index 1)
- `sample-tridiag`: Uses **single-document YAML**

### 5. **Variable Type (`var`)**

**For `hele-shaw`:**
- `var = "pressure"` - Solve for pressure field
- `var = "velocity"` - Solve for velocity field
- **Mutually exclusive** choices affecting which PDE is discretized

**For `sample-tridiag`:**
- `var` parameter is likely ignored (fixed tridiagonal structure)

## Summary Table

| Parameter | `sample-tridiag` | `hele-shaw` | Mutual Exclusivity |
|-----------|------------------|-------------|-------------------|
| `NQ_MATRIX` | ✓ Direct matrix size | ✓ Minimum size (may expand) | Different interpretation |
| `nx`, `ny` | ✓ Template only | ✓ **CFD mesh resolution** | Ignored vs. critical |
| `P_in`, `P_out` | ✗ Ignored | ✓ **Pressure BCs** | Not used vs. required |
| `U_top`, `U_bottom` | ✗ Ignored | ✓ **Velocity BCs** | Not used vs. required |
| `L`, `D` | ✗ Ignored | ✓ **Physical domain** | Not used vs. affects matrix |
| `mu` | ✗ Ignored | ✓ **Viscosity (κ impact!)** | Not used vs. critical |
| `rho` | ✗ Ignored | ✓ **Fluid density** | Not used vs. used |
| `var` | ? Likely ignored | ✓ **pressure/velocity** | Fixed vs. selectable |
| YAML format | Single document | **Two documents** | Structural difference |

## Regarding `nelem`

`nelem` was **not found** as a parameter in the qt02 codebase. It may be:
1. An internal variable in the ORNL circuit_HHL.py code (not exposed via TOML)
2. A parameter from a different version/fork
3. Related to "number of elements" in the finite element discretization (internal to hele-shaw)

The parameters you configure in TOML are limited to those in `input_vars.yaml` template, and `nelem` is not among them.

## List Expansion Feature

The qtlib workflow parser supports **list-valued parameters** that create combinatorial expansions:

```toml
[q3]
qc_shots = [100, 1000, 10000]
NQ_MATRIX = 3

# Expands to 3 separate cases:
# q3_0: qc_shots=100, NQ_MATRIX=3
# q3_1: qc_shots=1000, NQ_MATRIX=3  
# q3_2: qc_shots=10000, NQ_MATRIX=3
```

This allows systematic parameter sweeps where each case represents a unique, non-overlapping configuration.
