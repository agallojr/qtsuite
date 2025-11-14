import numpy as np
import scipy.linalg
import ffsim

# Create a simple one-body Hamiltonian
h1e = np.array([[1.0, 0.5], [0.5, -1.0]])
print("h1e (Hermitian):")
print(h1e)
print(f"Is Hermitian: {np.allclose(h1e, h1e.conj().T)}")
print(f"Is unitary: {np.allclose(h1e @ h1e.conj().T, np.eye(2))}")
print()

# Try passing it directly to OrbitalRotationJW
try:
    gate = ffsim.qiskit.OrbitalRotationJW(2, h1e)
    print("✓ Success with Hermitian matrix")
except ValueError as e:
    print(f"✗ Failed with Hermitian matrix: {e}")

# Compute time evolution operator
dt = 0.1
U = scipy.linalg.expm(-1j * dt * h1e)
print(f"\nU = exp(-i*dt*H):")
print(f"Is unitary: {np.allclose(U @ U.conj().T, np.eye(2))}")

try:
    gate = ffsim.qiskit.OrbitalRotationJW(2, U)
    print("✓ Success with time evolution operator")
except ValueError as e:
    print(f"✗ Failed with time evolution operator: {e}")

# Try with orbital rotation from diagonalization
eigvals, eigvecs = np.linalg.eigh(h1e)
print(f"\nOrbital rotation from diagonalization:")
print(f"Is unitary: {np.allclose(eigvecs @ eigvecs.conj().T, np.eye(2))}")

try:
    gate = ffsim.qiskit.OrbitalRotationJW(2, eigvecs)
    print("✓ Success with eigenvectors")
except ValueError as e:
    print(f"✗ Failed with eigenvectors: {e}")
