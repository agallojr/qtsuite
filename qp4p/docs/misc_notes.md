        - VQLS (and most variational quantum linear solvers) find the solution vector up to a global phase. This is fundamental to quantum mechanics - the state |ψ⟩ and e^(iφ)|ψ⟩ are physically indistinguishable. The algorithm minimizes the cost function which only depends on |⟨x|A†b⟩|², not the phase.
        - Construction method:
A = R·R^T + n·I

Where:

R is a random size × size matrix with entries drawn from standard normal distribution
R^T is the transpose of R
n·I is size times the identity matrix
Why this construction?
R·R^T produces a symmetric positive semi-definite matrix (all eigenvalues ≥ 0)
Adding n·I shifts all eigenvalues up by n, making them strictly positive
Result: symmetric positive definite (SPD) matrix
Properties guaranteed:
Always invertible (det(A) ≠ 0)
Well-conditioned - the eigenvalue shift prevents very small eigenvalues
Symmetric - A = A^T
Real eigenvalues - all positive
Condition number stays reasonable (typically 2-5 for small systems)


    + other
        - length of study, tools help
        - list of algorithms & use cases covered
        - error correction sweep
        - IBM Miami connectivity, connectivity screen shot
        - instructional goals
        - are the gs and ax=b examples comparable?
        - more pre-work cuz shorter class time
        - commonality between CFD & MAT - algorithms, software / tooling
        - full stack tool & lib zoo / icons
        - pennylane interop example
        - PyQrack interop example
        - Pennylane + Torch example
        - Classiq
        - real hardware access - IBM, IonQ, QuEra
        + VQE, (SQD, SKQD, HIVQE), Adapt-VQE eval 
            - analysis of similarities / differences
            - SKQD w Toño - bigger cases, HPC post-processing
            - Qunova NDA w ORNL
        - hpc cases

    + mod0 
        - slides
        - IBM steps
        - hello world example & show histogram == qubit ghz 2 which includes a display
        - qubit example - ghz
        - transpilation & execute lib - code dive
        - noise & backends, sims and remote
        - shots, sweeps, UQ
        - ./sweeper.sh mod0/input/qubits.toml --group ghz_noise_uq
    + mod1
        - slides
    + mod2 
        - slides
        - 1d nozzle code with existing libs
        - other CFD pre-reading?
        - ./sweeper.sh mod2/input/grovers.toml --group iteration_study
    + mod3 
        - slides
        - various gs and siam examples 
        - other MAT pre-reading?
        - homework mod3: https://pennylane.ai/qml/demos/tutorial_quantum_phase_transitions
    + topics
        - encoding - amplitude, basis, superdense, etc.
        - surface codes - mod1 
        - max cut with lattice
        - Hamiltonian simulation, Trotterization
        - eigensolvers - QPE, VQE, etc. - relationship to ground state problem
        - review Mike & Ike chapters
        - Ising model
        - portfolio optimization
        - QAOA (https://quantumai.google/cirq/experiments/qaoa/example_problems)
        - Grover's optimal iterations, variational optimal, shots estimates
        - error mitigation techniques https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques
        - teleportation