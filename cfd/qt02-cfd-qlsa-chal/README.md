***************************************************************************************
Source Code Origin
=====

ORNL provided code used in their Frontier training (https://github.com/olcf/hands-on-with-frontier). The pertinent HHL portion originated with the Winter Challenge 2025 (https://github.com/olcf/wciscc2025), and notably the key `linear_solvers` library used in the current code comes from (https://github.com/jw676/quantum_linear_solvers) which is a fork of our work (https://github.com/agallojr/quantum_linear_solvers) which we of course forked from them and updated it to work with Qiskit 2.0+, from its 0.x origins. We then forked hands-on-with-frontier - the subfolder which contains the QLSA code.

In order to orchestrate workflows and keep artifacts separated, we use some workflow tooling we developed (https://github.com/lwfm-proj/lwfm) which is inspired by our industrial experience (https://link.springer.com/chapter/10.1007/978-3-031-23606-8_16).

The ORNL code contains references to IonQ and IQM drivers. In the case of IQM, these are pinned to older version of Qiskit. We have a strong preference to stay on the tip of the major lib version trees like Qiskit, thus in our fork of the ORNL code we removed these dependencies. 


***************************************************************************************
Statevectors vs. Measurement-based
====

statevector: "a column vector within a complex vector space that completely describes the state of a quantum system. For a system of n qubits, the state vector is a 2^n-dimensional complex vector whose components represent the probability amplitudes for the system to be in each possible configuration."

The sample code (test_linear_solver.py) uses statevector-based HHL - barring perhaps changes from a random seed, it is going to produce the same result every time. HHL.solve() also uses a statevector with observables and computes an expectation value. (see linear_solvers/hhl.py)

You also cannot (in Qiskit) introduce noise into a statevector-based simulation.

We modified the ORNL code to use measurement-based HHL - to make it accept shot counts, which we expect to impact the fidelity of the results. To do this, we inject qubit measurements into the circuit which is produced by the ORNL code. 

Because we produce the circuit in one runtime sandbox (aka site), pinned to its own dependencies, and run it on another (again with its own dependencies), we use QPY to pass the circuit between them. This might not be useful (t.b.d.) if we wanted to move outside of Qiskit. We find out that the measurement circuit must be added in the runtime site, set to the specific quantum backend, and we transpile for that backend in that runtime sandbox site.


***************************************************************************************
To Do
=====

- what is fidelity of current classical method in use?
- benchmark statement
- strong vs weak scaling


***************************************************************************************
