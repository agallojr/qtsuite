"""
Test 01: defaults
"""

#pylint: disable=invalid-name

import time
import numpy as np
from scipy.sparse import diags

from linear_solvers import NumPyLinearSolver, HHL

# from lwfm.base.JobDefn import JobDefn
# from lwfm.base.Workflow import Workflow
# from lwfm.midware.LwfManager import lwfManager



if __name__ == '__main__':
    # Generate matrix and vector
    # We use a sample tridiagonal system. It's 2x2 version is:
    # matrix = np.array([ [1, -1/3], [-1/3, 1] ])
    # vector = np.array([1, 0])
    n_qubits_matrix = 2
    MATRIX_SIZE = 2 ** n_qubits_matrix
    # entries of the tridiagonal Toeplitz symmetric matrix
    a = 1
    b = -1/3
    matrix = diags([b, a, b],
                [-1, 0, 1],
                shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    vector = np.array([1] + [0]*(MATRIX_SIZE - 1))


    # ============
    # Setup HHL solver
    hhl = HHL(1e-3)

    # ============
    # Solutions
    print('======================')
    # Classical
    t = time.time()
    classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
    elpsdt = time.time() - t
    print(f'Time elapsed for classical:\n{int(elpsdt/60)} min {elpsdt%60:.2f} sec')
    # HHL
    t = time.time()
    hhl_solution = hhl.solve(matrix, vector)
    elpsdt = time.time() - t
    print(f'Time elapsed for HHL:\n{int(elpsdt/60)} min {elpsdt%60:.2f} sec')

    # ============
    # Circuits
    print('======================')
    print('HHL circuit:')
    print(hhl_solution.state)

    # ============
    # Comparing the observable - Euclidean norm
    print('======================')
    print(f'Euclidean norm classical:\n{classical_solution.euclidean_norm}')
    print(f'Euclidean norm HHL:\n{hhl_solution.euclidean_norm} (diff (%): {np.abs(classical_solution.euclidean_norm-hhl_solution.euclidean_norm)*100/classical_solution.euclidean_norm:1.3e})')

    # ============
    # Comparing the solution vectors component-wise
    print('======================')
    from qiskit.quantum_info import Statevector
    def get_solution_vector(solution, nstate):
        """
        Extracts and normalizes simulated state vector
        from LinearSolverResult.
        """
        # solution_vector = Statevector(solution.state).data[-nstate:].real
        temp = Statevector(solution.state)
        ID = np.where(np.abs(temp.data[:].real)<1e-10)[0]
        A = temp.data[:]
        A[ID] = 0+0j
        B = temp.data[:].real
        B[ID] = 0
        # print(f'# of elements in solution vector: {len(B)}')
        istart = int(len(B)/2)
        solution_vector = temp.data[istart:istart+nstate].real
        norm = solution.euclidean_norm
        return norm * solution_vector / np.linalg.norm(solution_vector)

    print(f'Classical solution vector:\n{classical_solution.state}')
    solvec_hhl = get_solution_vector(hhl_solution, MATRIX_SIZE)
    print(f'HHL solution vector:\n{solvec_hhl}')
    print(f'diff (%):\n{np.abs(classical_solution.state-solvec_hhl)*100/classical_solution.state}')


    # # use lwfm to run this circuit on ibm quantum cloud &/or their simulators
    # site = lwfManager.getSite("ibm-quantum-venv")

    # # let's try it on a simulator of ibm_brisbane 
    # backend = "ibm_brisbane_aer"

    # # define a workflow - this workflow is just one job, but this is an illustration...
    # wf = Workflow()
    # wf.setName(f"hhl:{backend}")
    # wf.setDescription(f"hhl example on {backend}")
    # wf.setProps({"backend": backend, "cuzReason": "for giggles"})

    # # define the job - we can use qasm3 to interop between vendor backends (or use raw qiskit)
    # jobDefn = JobDefn(dumps(getCircuit()), JobDefn.ENTRY_TYPE_STRING, {"format": "qasm3"})

    # # some args for this run
    # runArgs = {
    #     "shots": 1024,                  # number of runs of the circuit
    #     "optimization_level": 3         # agressive transpiler optimization (values: 0-3)
    # }
    # runArgs["computeType"] = backend
    # status = site.getRunDriver().submit(jobDefn, wf, runArgs["computeType"], runArgs)
    
    # # wait synchronously for the job to finish (lwfm also does async)
    # status = lwfManager.wait(status.getJobId())
    # print(f"Job {status.getJobId()} finished with status {status.getStatus()}")
