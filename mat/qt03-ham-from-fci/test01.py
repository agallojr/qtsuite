"""
hamiltonian from fci
"""

#pylint: disable=protected-access

import numpy as np
from pyscf import gto, scf, fci, ao2mo
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Estimator
from scipy.optimize import minimize

# 1. Create molecule and get integrals
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol).run()

# 2. Get integrals directly from mean-field calculation
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
# 3. Transform 2-electron integrals to MO basis
h2e = ao2mo.kernel(mol, mf.mo_coeff)
ecore = mf.energy_nuc()
norb = mf.mo_coeff.shape[1]
nelec = mol.nelec

# 4. Construct FCI Hamiltonian matrix
# For small systems, you can construct the dense matrix directly
# Note: For large systems, this is not feasible.
fci_solver = fci.FCI(mf)
indices, hamiltonian_matrix = fci_solver.pspace(h1e, h2e, norb, nelec)
hamiltonian_matrix = hamiltonian_matrix + np.diag([ecore] * hamiltonian_matrix.shape[0])

# Print the Hamiltonian matrix
print("FCI Hamiltonian Matrix:")
print(hamiltonian_matrix)

# 5. Convert to weighted Pauli operator
hamil_qop = SparsePauliOp.from_operator(hamiltonian_matrix)

print("\nWeighted Pauli Operator:")
print(hamil_qop)

# 6. Create quantum circuit for the Hamiltonian
# Determine number of qubits needed
num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
print(f"\nNumber of qubits needed: {num_qubits}")

# Create a variational ansatz circuit
ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=2, entanglement='linear')
print("\nVariational Ansatz Circuit:")
print(ansatz)

# Create a simple state preparation circuit (Hartree-Fock initial state)
hf_circuit = QuantumCircuit(num_qubits)
# For H2 molecule, put electrons in lowest energy orbitals
# Assuming 2 electrons in 2 qubits (spin-up and spin-down)
if num_qubits >= 2:
    hf_circuit.x(0)  # Spin-up electron
    hf_circuit.x(1)  # Spin-down electron

print("\nHartree-Fock Initial State Circuit:")
print(hf_circuit)

# Combine initial state + ansatz for VQE
vqe_circuit = hf_circuit.compose(ansatz)
vqe_circuit = vqe_circuit.decompose()
print("\nComplete VQE Circuit (HF + Ansatz):")
print(vqe_circuit)
print(f"\nQubits: {num_qubits} Number of gates: {vqe_circuit.count_ops()} "
      f"depth {vqe_circuit.depth()}")

# 7. Run VQE to find ground state
print("\n" + "="*50)
print("RUNNING VQE OPTIMIZATION")
print("="*50)

# Create estimator for expectation value calculations
estimator = Estimator()

# Define cost function for VQE
def cost_function(params):
    """Calculate expectation value <psi(params)|H|psi(params)>"""
    # Create a copy of the ansatz and assign parameters
    param_dict = dict(zip(ansatz.parameters, params))
    bound_circuit = ansatz.assign_parameters(param_dict)
    # Combine with initial state
    full_circuit = hf_circuit.compose(bound_circuit)
    
    # Calculate expectation value
    job = estimator.run(full_circuit, hamil_qop)
    result = job.result()
    energy = result.values[0]
    
    return energy

# Initialize parameters randomly
num_params = ansatz.num_parameters
initial_params = np.random.uniform(0, 2*np.pi, num_params)
print(f"Number of parameters: {num_params}")
print(f"Initial parameters: {initial_params}")

# Run optimization
print("\nStarting optimization...")
result = minimize(cost_function, initial_params, method='COBYLA', 
                 options={'maxiter': 1000, 'disp': True})

print("\n" + "="*50)
print("VQE RESULTS")
print("="*50)
print(f"Optimization success: {result.success}")
print(f"VQE ground state energy: {result.fun:.8f} Hartree")
print(f"SCF energy (reference): {mf.energy_tot():.8f} Hartree")
print(f"Energy difference: {abs(result.fun - mf.energy_tot()):.8f} Hartree")
print(f"Optimal parameters: {result.x}")

# Get exact ground state energy from Hamiltonian diagonalization
eigenvalues = np.linalg.eigvals(hamiltonian_matrix)
exact_ground_energy = np.min(eigenvalues)
print(f"Exact ground state energy: {exact_ground_energy:.8f} Hartree")
print(f"VQE error vs exact: {abs(result.fun - exact_ground_energy):.8f} Hartree")
