import numpy as np
import skqd_helpers
import scipy.linalg
from qiskit import QuantumRegister, QuantumCircuit

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

print("Testing WITHOUT transpose:")
h1e_momentum1, h2e_momentum1 = skqd_helpers.rotated(h1e, h2e, orbital_rotation)
H_test1 = (
    np.round(np.array(h1e_momentum1, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_momentum1, dtype=np.float64), decimals=12),
)

qreg = QuantumRegister(2 * NUM_ORBS, 'q')
qc = QuantumCircuit(qreg)

try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_test1, IMPURITY_INDEX, NUM_ORBS):
        qc.append(instruction)
    print("  ✓ Success without transpose!")
except Exception as e:
    print(f"  ✗ Error without transpose: {e}")

print("\nTesting WITH transpose:")
h1e_momentum2, h2e_momentum2 = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)
H_test2 = (
    np.round(np.array(h1e_momentum2, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_momentum2, dtype=np.float64), decimals=12),
)

qc2 = QuantumCircuit(qreg)

try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_test2, IMPURITY_INDEX, NUM_ORBS):
        qc2.append(instruction)
    print("  ✓ Success with transpose!")
except Exception as e:
    print(f"  ✗ Error with transpose: {e}")
