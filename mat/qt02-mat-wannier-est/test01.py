"""
Estimate quantum sizing information about a Wannier Hamiltonian derived from JARVIS data.
"""
import warnings

from pyparsing import pprint
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import pprint as pp

from jarvis.db.figshare import get_wann_electron, get_hk_tb
from qiskit.aqua.operators import MatrixOperator, op_converter
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
import numpy as np
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance

from get_wf_args import get_cases_args

def variational_evolution(hamil_op, num_qubits, maxiter=500) -> float:
    optimizer = COBYLA(maxiter=maxiter)   

    # RYRZ is closest to EfficientSU2 in qiskit-aqua 0.7.5
    # depth controls circuit depth, entanglement controls connectivity
    var_form = RYRZ(num_qubits=num_qubits, depth=3, entanglement='full')
    vqe = VQE(hamil_op, var_form, optimizer,
        quantum_instance=QuantumInstance(backend=BasicAer.get_backend('statevector_simulator')))
    result = vqe.run()  
    return result["eigenvalue"].real


if __name__ == "__main__":

    # Load configuration from TOML file
    casesArgs = get_cases_args()
    globalArgs = casesArgs["global"]
    
    for caseId, caseArgs in ((k, v) for k, v in casesArgs.items() if k != "global"):
        caseArgs.update(globalArgs)

        kx = caseArgs["kx"]
        ky = caseArgs["ky"]
        kz = caseArgs["kz"]
        jid = caseArgs["jid"]
        var_form_maxiter = caseArgs["var_form_maxiter"]

        k = [kx, ky, kz]

        # ******************************************************************************
        # original code

        # get the Hamiltonian matrix from JARVIS
        w,ef,atoms=get_wann_electron(jid=jid)
        hk=get_hk_tb(w=w,k=k)
        hamil_mat=MatrixOperator(hk)
        # build up the Paulis
        hamil_qop = op_converter.to_weighted_pauli_operator(hamil_mat)
    
        # ******************************************************************************
        # added code

        # convert the Hamiltonian to a circuit, using evolve method (creates Trotterization circuit)
        qr = QuantumRegister(hamil_qop.num_qubits)
        circuit = QuantumCircuit(qr)
        evo_time = 1.0              # evolution time parameter
        evo_time_slices = 1000      # number of time slices
        evolved_circuit = hamil_qop.evolve(None, evo_time, evo_time_slices, quantum_registers=qr)
        decomposed_evolved_circuit = evolved_circuit.decompose()

        # Get exact evolution using matrix exponential
        hamil_matrix = hamil_mat.dense_matrix
        exact_unitary = np.exp(-1j * hamil_matrix * evo_time)
        
        # Calculate fidelity of Trotterized circuit
        circuit_unitary = Operator(evolved_circuit).data
        trotter_fidelity = np.abs(np.trace(circuit_unitary.conj().T @ exact_unitary)) / \
            circuit_unitary.shape[0]

        # Get exact ground state energy
        eigenvalues = np.linalg.eigvalsh(hamil_matrix)
        exact_ground_energy = eigenvalues[0]
        
        # Compare to variational evolution
        variational_energy = variational_evolution(hamil_qop, hamil_qop.num_qubits, var_form_maxiter)
        energy_error = abs(variational_energy - exact_ground_energy)
        energy_fidelity = 1.0 - (energy_error / abs(exact_ground_energy)) \
            if exact_ground_energy != 0 else 1.0

        print("******")
        print(f"Case ID: {caseId}")
        print(f"JID: {jid}")
        #pp.pprint(f"w: {w.to_dict()}")
        print(f"ef: {ef}")
        print("atoms structure:")
        pp.pprint(atoms.to_dict())
        print(f"k: {k}")
        print(f"qubits: for Hamiltonian: {hamil_qop.num_qubits}, "
            f"in circuit: {decomposed_evolved_circuit.num_qubits}")
        print(f"hamil_matrix shape:\t {hamil_matrix.shape}")
        print(f"circuit_unitary shape:\t {circuit_unitary.shape}")
        print(f"exact_unitary shape:\t {exact_unitary.shape}")
        print(f"Trotter:")
        print(f"  Steps: {evo_time_slices} | Depth: {decomposed_evolved_circuit.depth()} | "
            f"Gates: {sum(decomposed_evolved_circuit.count_ops().values())} | Fidelity: "
            f"{trotter_fidelity:.6f}")
        print(f"Variational:")
        print(f"  Exact ground energy: {exact_ground_energy:.6f}")
        print(f"  VQE energy:          {variational_energy:.6f}")
        print(f"  Energy fidelity:     {energy_fidelity:.6f}")
        print("******")

