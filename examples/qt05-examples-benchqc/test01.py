"""
This script is designed to set up a quantum circuit for simulating the electronic structure
of a molecule using Qiskit and Jarvis libraries.

Some commented out parts could be used to get the classical solution using numpy.

There are also several loops within loops where one could iterate over a number of 
different parameters.
"""

#pylint: disable=missing-function-docstring, redefined-outer-name, unused-argument
#pylint: disable=wrong-import-position

import warnings

# Suppress all warnings before importing Qiskit Nature modules
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress Qiskit Nature specific warnings with comprehensive pattern matching
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*qiskit_nature.*")
warnings.filterwarnings("ignore", module="qiskit_nature.*")
# Catch the specific pattern from your error message
warnings.filterwarnings("ignore", message=".*algorithms.excited_state_solvers.*deprecated.*")
warnings.filterwarnings("ignore", message=".*algorithms.pes_sampler.*deprecated.*")

import numpy as np

# Importing Qiskit and Jarvis libraries
from qiskit import Aer
# from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.drivers.second_quantization import MethodType
from qiskit_nature.transformers.second_quantization.electronic.active_space_transformer \
    import ActiveSpaceTransformer
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver

from jarvis.core.circuits import QuantumCircuitLibrary

import matplotlib.pyplot as plt


def make_diatomic_system(element='Al', bond_distance=1.2):
    pos = [(element, [0, 0, 0]), (element, [0, 0, bond_distance])]
    return pos


def get_qubit_op(
    molecule,
    basis='sto3g',
    functional='lda',
    method= MethodType.RKS,
    driver_type= ElectronicStructureDriverType.PYSCF,
    converter= JordanWignerMapper()):

    driver=ElectronicStructureMoleculeDriver(molecule, basis, method,driver_type)
    properties = driver.run()
    problem = ElectronicStructureProblem(driver)

    second_q_ops = problem.second_q_ops()
    if not second_q_ops:
        raise ValueError("No second quantized operators found in the problem.")
    # second_q_ops is a ListOrDict, convert to list using values()
    # second_q_ops_list = list(second_q_ops.values())  # type: ignore
    # hamiltonian = second_q_ops_list[0]

    # numpy_solver = NumPyMinimumEigensolver()
    tmp = properties.get_property('ParticleNumber')
    if tmp is None:
        raise ValueError("ParticleNumber property not found in the properties.")
    alpha_occ=tmp.occupation_alpha
    beta_occ=tmp.occupation_beta

    mo_considered=3
    #active and inactive space has to be even, non-magnetic

    first_index = min(np.where(alpha_occ<1)[0][0],np.where(beta_occ<1)[0][0])

    orb_act = np.arange(first_index-mo_considered,first_index)+1

    transformer= ActiveSpaceTransformer(num_electrons=mo_considered+1,
                                        num_molecular_orbitals=len(orb_act),
                                        active_orbitals=orb_act)
    problem_reduced = ElectronicStructureProblem(driver, [transformer])
    second_q_ops_reduced = problem_reduced.second_q_ops()

    # second_q_ops_reduced is a ListOrDict, convert to list using values()
    second_q_ops_reduced_list = list(second_q_ops_reduced.values())  # type: ignore
    hamiltonian_reduced = second_q_ops_reduced_list[0]
    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper)
    qubit_op = converter.convert(hamiltonian_reduced)

    res1={}
    res1['qubit_op']=qubit_op
    res1['converter']=converter
    res1['problem_reduced']=problem_reduced

    return res1 #qubit_op, converter, problem_reduced


def get_energy(optimizer=None,device=None,qubit_op=None,seed=None):
    seed = 42
    counts = []
    values = []


    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    algorithm_globals.random_seed = seed
    print('device',device)
    print('seed',seed)

    qi = QuantumInstance(device, seed_transpiler=seed, seed_simulator=seed)
    n_qubits = qubit_op.num_qubits # type: ignore
    ansatz = QuantumCircuitLibrary(n_qubits=n_qubits, reps=1).circuit6()
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=qi,callback=store_intermediate_result)
    if qubit_op is None:
        raise ValueError("qubit_op is None. Cannot compute minimum eigenvalue.")
    result = vqe.compute_minimum_eigenvalue(operator=qubit_op)


    eigenvalue = result.eigenvalue
    # return eigenvalue, vqe, qi

    res={}
    res['eigenvalue']=eigenvalue
    res['vqe']=vqe
    res['qi']=qi
    return res


elements=['Al']
basis=['sto3g']
method=[MethodType.RKS]
#driver_type=[]
optimizer=[SLSQP(maxiter=1000)]
#converter=[JordanWignerMapper()]
functionals=['lda']
devices=[Aer.get_backend('statevector_simulator')]
mem={}

bond_distances = [1.0, 1.5, 2.0, 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0]
energies = []


for bond_distance in bond_distances:
    # Pass bond_distance to the function
    positions = make_diatomic_system(bond_distance=bond_distance)
    molecule = Molecule(geometry=positions, charge=0, multiplicity=1)

    for i in elements:
        for j in basis:
            for k in method:
            #for l in driver_type:
                for m in optimizer:
                    #for n in converter:
                    for o in functionals:
                        for p in devices:
                            print(bond_distance,i,j,k,m,o,p)

                            res1=get_qubit_op(molecule=molecule,basis=j,functional=o,method=k)
                            print(f"res1: {res1}")
                            res=get_energy(optimizer=m,device=p,qubit_op=res1['qubit_op'],seed=42)
                            print(f'Bond distance {bond_distance}: {res["eigenvalue"]}')

                            #print(res['eigenvalue'])

                            #GroundStateEigensolver
                            solver =GroundStateEigensolver(res1['converter'], res['vqe'])
                            result=solver.solve(res1['problem_reduced'])
                            #print(result)

                            # Collect bond distance and energy
                            energies.append(result.total_energies[0].real) #type: ignore

                            mem[bond_distance,i,j,k,m,o,p]= {
                                'eigenvalue': res['eigenvalue'],
                                'vqe': res['vqe'],
                                'qi': res['qi'],
                                'converter': res1['converter'],
                                'problem_reduced': res1['problem_reduced']
                            }


# Plotting the results
plt.plot(bond_distances, energies, marker='o')
plt.xlabel('Bond Distance (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy vs. Bond Distance')
plt.grid(True)
plt.show()
