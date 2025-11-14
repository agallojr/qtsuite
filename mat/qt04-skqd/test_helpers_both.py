import numpy as np
import skqd_helpers
from qiskit import QuantumRegister, QuantumCircuit

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE
IMPURITY_INDEX = (NUM_ORBS - 1) // 2
DT = 0.15

# Get the position-basis Hamiltonian
h1e_pos, h2e_pos = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

# Test with position-basis Hamiltonian
H_pos = (
    np.round(np.array(h1e_pos, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_pos, dtype=np.float64), decimals=12),
)

qreg = QuantumRegister(2 * NUM_ORBS, 'q')
qc = QuantumCircuit(qreg)

print("Test with POSITION-BASIS Hamiltonian:")
try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_pos, IMPURITY_INDEX, NUM_ORBS):
        qc.append(instruction)
    print("  ✓ SUCCESS with position-basis Hamiltonian!")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test with momentum-basis (no transpose)
orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)
h1e_mom1, h2e_mom1 = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation)
H_mom1 = (
    np.round(np.array(h1e_mom1, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_mom1, dtype=np.float64), decimals=12),
)

qc2 = QuantumCircuit(qreg)
print("\nTest with MOMENTUM-BASIS (no .T):")
try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_mom1, IMPURITY_INDEX, NUM_ORBS):
        qc2.append(instruction)
    print("  ✓ SUCCESS!")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test with momentum-basis (with transpose)
h1e_mom2, h2e_mom2 = skqd_helpers.rotated(h1e_pos, h2e_pos, orbital_rotation.T)
H_mom2 = (
    np.round(np.array(h1e_mom2, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_mom2, dtype=np.float64), decimals=12),
)

qc3 = QuantumCircuit(qreg)
print("\nTest with MOMENTUM-BASIS (with .T):")
try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_mom2, IMPURITY_INDEX, NUM_ORBS):
        qc3.append(instruction)
    print("  ✓ SUCCESS!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
