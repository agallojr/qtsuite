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

# Create a more expressive variational ansatz circuit
ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=3, entanglement='full')
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

# Create template VQE circuit for visualization (parameters unbound)
template_circuit = hf_circuit.compose(ansatz)
template_circuit = template_circuit.decompose()
print("\nVQE Circuit Template (parameters unbound):")
print(template_circuit)
print(f"\nQubits: {num_qubits} Number of gates: {template_circuit.count_ops()} "
      f"depth {template_circuit.depth()}")
print("Note: Actual VQE uses this template with dynamically bound parameters")

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

# Initialize parameters more intelligently
num_params = ansatz.num_parameters
# Start with small random values near zero for better convergence
initial_params = np.random.uniform(-0.1, 0.1, num_params)
print(f"Number of parameters: {num_params}")
print(f"Initial parameters: {initial_params}")

# Run optimization with multiple attempts
print("\nStarting optimization...")
best_result = None
best_energy = float('inf')

for attempt in range(3):
    print(f"\nOptimization attempt {attempt + 1}/3...")
    if attempt > 0:
        # Try different starting points
        initial_params = np.random.uniform(-0.5, 0.5, num_params)
    
    result = minimize(cost_function, initial_params, method='COBYLA', 
                     options={'maxiter': 2000, 'disp': False})
    
    if result.fun < best_energy:
        best_energy = result.fun
        best_result = result
        print(f"New best energy: {best_energy:.8f} Hartree")

result = best_result

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

# Chemical accuracy analysis
vqe_error = abs(result.fun - exact_ground_energy)
chemical_accuracy_hartree = 0.0015936  # 1 kcal/mol in Hartree
chemical_accuracy_kcal = vqe_error * 627.509  # Convert Hartree to kcal/mol

print(f"VQE error vs exact: {vqe_error:.8f} Hartree")
print(f"VQE error vs exact: {chemical_accuracy_kcal:.4f} kcal/mol")
print(f"Chemical accuracy threshold: {chemical_accuracy_hartree:.6f} Hartree (1.0 kcal/mol)")

if vqe_error <= chemical_accuracy_hartree:
    print("✓ VQE result is within chemical accuracy!")
else:
    accuracy_ratio = vqe_error / chemical_accuracy_hartree
    print(f"✗ VQE error is {accuracy_ratio:.2f}x larger than chemical accuracy")
