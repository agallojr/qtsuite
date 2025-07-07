"""
Test 01: defaults
"""

#pylint: disable=invalid-name

import time
import numpy as np
from scipy.sparse import diags

from linear_solvers import HHL


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
    # # Classical
    # t = time.time()
    # classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
    # elpsdt = time.time() - t
    # print(f'Time elapsed for classical:\n{int(elpsdt/60)} min {elpsdt%60:.2f} sec')
    # HHL
    t = time.time()
    hhl_solution = hhl.solve(matrix, vector)
    elpsdt = time.time() - t
    print(f'Time elapsed for HHL:\n{int(elpsdt/60)} min {elpsdt%60:.2f} sec')
