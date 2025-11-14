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

# Get the Hamiltonian
h1e, h2e = skqd_helpers.siam_hamiltonian(
    NUM_ORBS, HOPPING, ONSITE, HYBRIDIZATION, CHEMICAL_POTENTIAL
)

orbital_rotation = skqd_helpers.momentum_basis(NUM_ORBS)
h1e_momentum, h2e_momentum = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)

# Round like the grader does
H_ref = (
    np.round(np.array(h1e_momentum, dtype=np.float64), decimals=12),
    np.round(np.array(h2e_momentum, dtype=np.float64), decimals=12),
)

print("h1e_momentum shape:", H_ref[0].shape)
print("h1e_momentum is Hermitian:", np.allclose(H_ref[0], H_ref[0].conj().T))
print("h1e_momentum dtype:", H_ref[0].dtype)

# Try to create a Trotter step
qreg = QuantumRegister(2 * NUM_ORBS, 'q')
qc = QuantumCircuit(qreg)

print("\nTrying to create trotter_step...")
try:
    for instruction in skqd_helpers.trotter_step(qreg, DT, H_ref, IMPURITY_INDEX, NUM_ORBS):
        print(f"  Instruction: {instruction.operation.name}")
        qc.append(instruction)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    
    # Try with time-evolved operator
    print("\nTrying with expm(-i*dt*H)...")
    import scipy.linalg
    U = scipy.linalg.expm(-1j * DT * H_ref[0])
    print("U is unitary:", np.allclose(U @ U.conj().T, np.eye(NUM_ORBS)))
