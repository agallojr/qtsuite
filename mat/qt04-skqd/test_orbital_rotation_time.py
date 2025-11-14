import numpy as np
import scipy.linalg
import ffsim
import skqd_helpers
from qiskit import QuantumCircuit, QuantumRegister

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE
IMPURITY_INDEX = (NUM_ORBS - 1) // 2
DT = 0.15

# Get the Hamiltonian
h1e, h2e = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)
h1e_mom, h2e_mom = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)

H_ref = (
    np.round(np.array(h1e_mom, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_mom, dtype=np.float64), decimals=12),
)

# Try to manually compute what trotter_step should produce
print("Option 1: Pass exp(-i*t*H) to OrbitalRotationJW")
U_time_evolution = scipy.linalg.expm(-1j * DT * H_ref[0])
print(f"  Is unitary: {np.allclose(U_time_evolution @ U_time_evolution.conj().T, np.eye(NUM_ORBS))}")

qreg = QuantumRegister(2 * NUM_ORBS, 'q')
qc = QuantumCircuit(qreg)

try:
    gate = ffsim.qiskit.OrbitalRotationJW(NUM_ORBS, U_time_evolution)
    qc.append(gate, qreg)
    print("  ✓ Success!")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Option 2: Check if OrbitalRotationJW has a time parameter
print("\nOption 2: Check OrbitalRotationJW signature for time parameter")
import inspect
sig = inspect.signature(ffsim.qiskit.OrbitalRotationJW.__init__)
print(f"  Parameters: {list(sig.parameters.keys())}")
