import numpy as np
import scipy.linalg
import skqd_helpers

# Grader constants
NUM_ORBS = 10
HOPPING = 1.0
ONSITE = 5
HYBRIDIZATION = 1.0
CHEMICAL_POTENTIAL = -0.5 * ONSITE
DT = 0.15

# Get Hamiltonian
h1e, h2e = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)
h1e_momentum, h2e_momentum = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)

print("Original h1e_momentum:")
print(f"  Is Hermitian: {np.allclose(h1e_momentum, h1e_momentum.conj().T)}")
print(f"  Is Unitary: {np.allclose(h1e_momentum @ h1e_momentum.conj().T, np.eye(NUM_ORBS))}")
print()

# Convert to unitary time evolution operator
h1e_unitary = scipy.linalg.expm(-1j * DT * h1e_momentum)

print("Time evolution operator U = exp(-i*dt*H):")
print(f"  Is Hermitian: {np.allclose(h1e_unitary, h1e_unitary.conj().T)}")
print(f"  Is Unitary: {np.allclose(h1e_unitary @ h1e_unitary.conj().T, np.eye(NUM_ORBS))}")
print()

# Modified Hamiltonian tuple with unitary h1e
hamiltonian_unitary = (h1e_unitary, h2e_momentum)

print("If we pass this modified Hamiltonian to trotter_step:")
print(f"  hamiltonian[0] is unitary: {np.allclose(hamiltonian_unitary[0] @ hamiltonian_unitary[0].conj().T, np.eye(NUM_ORBS))}")
print()

# Test with skqd_helpers.trotter_step
from qiskit import QuantumRegister, QuantumCircuit
IMPURITY_INDEX = (NUM_ORBS - 1) // 2

qreg = QuantumRegister(2 * NUM_ORBS, 'q')
qc = QuantumCircuit(qreg)

H_rounded_unitary = (
    np.round(np.array(hamiltonian_unitary[0], dtype=np.complex128), decimals=12),
    np.round(np.array(hamiltonian_unitary[1], dtype=np.float64), decimals=12),
)

print("Testing with skqd_helpers.trotter_step:")
try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_rounded_unitary, IMPURITY_INDEX, NUM_ORBS):
        qc.append(instruction)
    print("  ✓ SUCCESS with unitary h1e!")
    print(f"  Circuit has {len(qc)} gates")
except Exception as e:
    print(f"  ✗ Failed: {e}")
