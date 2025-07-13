"""
Test 02: show how to run the ORNL WCISCC2025 in piece parts, separating out the 
circuit construction from the circuit execution. This will let us exercise their code
more broadly.
"""

from lwfm.base.JobDefn import JobDefn
from lwfm.base.Workflow import Workflow
from lwfm.midware.LwfManager import lwfManager


if __name__ == '__main__':

    site = lwfManager.getSite("local")

    # define a workflow, give it some metadata
    wf = Workflow()
    wf.setName("qt01_examples_wciscc2025.test_02")
    wf.setDescription("running the ORNL WCISCC2025 in 2 parts - circuit build & run")
    wf.setProps({"cuzReason": "for giggles"})

    # Generate matrix and vector
    # We use a sample tridiagonal system. It's 2x2 version is:
    # matrix = np.array([ [1, -1/3], [-1/3, 1] ])
    # vector = np.array([1, 0])
    n_qubits_matrix = args.NQ_MATRIX
    MATRIX_SIZE = 2 ** n_qubits_matrix
    # entries of the tridiagonal Toeplitz symmetric matrix
    a = 1
    b = -1/3
    matrix = diags([b, a, b],
                [-1, 0, 1],
                shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    vector = np.array([1] + [0]*(MATRIX_SIZE - 1))

    # ============
    # Select backend: Using different simulators (default in `linear_solvers`
    # is statevector simulation)
    backend = AerSimulator(method='statevector')
    backend.set_options(precision='single')

    # ============
    # Setup HHL solver
    hhl = HHL(1e-3, quantum_instance=backend)
    print(f'Simulator: {backend}')

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
