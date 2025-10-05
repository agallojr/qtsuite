"""
hamiltonian from fci
"""

#pylint: disable=protected-access, invalid-name

import numpy as np
from pyscf import gto, scf, fci, ao2mo
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

# Create molecule and get integrals.
# This could be read in from an FCI file - here's an elaborate example:
#
# # Read in molecule from disk and mine it for integrals
# molecule_scf = tools.fcidump.to_scf(caseArgs['input_fcidump'])
# print("Molecule read from file.")
# # Core Hamiltonian representing the single-electron integrals
# core_hamiltonian = molecule_scf.get_hcore()
# print("Core Hamiltonian created.")
# # Electron repulsion integrals representing the two-electron integrals
# electron_repulsion_integrals = ao2mo.restore(1, molecule_scf._eri,
#     caseArgs['num_orbitals'])
#
# Keeping it simple for now with H2
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol).run()

# Get integrals directly from mean-field calculation
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

# Transform 2-electron integrals to MO basis
h2e = ao2mo.kernel(mol, mf.mo_coeff)
ecore = mf.energy_nuc()
norb = mf.mo_coeff.shape[1]
nelec = mol.nelec

# Construct full FCI Hamiltonian matrix
fci_solver = fci.FCI(mf)

# Get exact FCI energy for comparison
fci_energy = fci_solver.kernel()[0]
print(f"Exact FCI energy from PySCF: {fci_energy:.8f} Hartree")

# Build Hamiltonian matrix in full CI space
# For H2 with 2 orbitals and 2 electrons, we have 4 determinants (singlets only for RHF)
# Use pspace with large np to get full space
indices, hamiltonian_matrix = fci_solver.pspace(h1e, h2e, norb, nelec, np=99)
hamiltonian_matrix = hamiltonian_matrix + np.diag([ecore] * hamiltonian_matrix.shape[0])
print(f"Hamiltonian matrix shape: {hamiltonian_matrix.shape}")

# Convert to Pauli operator
hamil_qop = SparsePauliOp.from_operator(hamiltonian_matrix)
exact_ground = fci_energy

# VQE Setup
num_qubits = hamil_qop.num_qubits
print(f"Hamiltonian requires {num_qubits} qubits")

# Use simpler ansatz
ansatz = EfficientSU2(num_qubits, reps=2, entanglement='linear')

# VQE cost function using statevector simulation
def vqe_cost(params):
    """Compute <ψ(θ)|H|ψ(θ)> using statevector"""
    bound_circuit = ansatz.assign_parameters(params)
    psi = Statevector.from_instruction(bound_circuit)
    energy = psi.expectation_value(hamil_qop).real
    return energy

# Multiple optimization attempts with different starting points
best_result = None
best_energy = float('inf')

for attempt in range(5):
    # Random initial parameters
    initial_params = np.random.uniform(-0.1, 0.1, ansatz.num_parameters)

    # Optimize with COBYLA
    result = minimize(vqe_cost, initial_params, method='COBYLA', 
                     options={'maxiter': 2000, 'tol': 1e-8})

    if result.fun < best_energy:
        best_energy = result.fun
        best_result = result
        print(f"  Attempt {attempt+1}: Energy = {result.fun:.8f} Hartree")

result = best_result
print(f"\nBest result after 5 attempts: {result.fun:.8f} Hartree")

# Results
print(f"\nVQE ground state energy: {result.fun:.8f} Hartree")
print(f"Exact FCI energy:        {exact_ground:.8f} Hartree")
print(f"SCF reference energy:    {mf.energy_tot():.8f} Hartree")
vqe_error = abs(result.fun - exact_ground)
chemical_accuracy = 0.0015936
print(f"VQE error:               {vqe_error:.8f} Hartree ({vqe_error/chemical_accuracy:.2f}x chem. accuracy)")
