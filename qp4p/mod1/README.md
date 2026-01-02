# Quantum Programming for Programmers (QP4P) - Module 1

## ▸ Module 1: Prerequisites

- Attend or otherwise consume QP4P Module 0. Review the foundational concepts from Module 0 to ensure familiarity with quantum computing basics and Python programming.

- Intermediate-level Python programming skills, including familiarity with functions, classes, libraries such as numpy and matplotlib, and the use of virtual environments or other means to manage dependencies.

- A working Python environment set to the libraries published for the class. We recommend the use of uv (https://docs.astral.sh/uv/). We have published a specification file (see ../pyproject.toml) that defines the required dependencies. Sample code will be provided to verify the environment. Please come to class with a working environment as we will not have time to troubleshoot environment issues during class.

- Run the Module 0 examples. 

- Read "Quantum Computation and Quantum Information" by Mike & Ike, Chapters 1-4, 7, or seek alternative introductory materials to build foundational understanding on topics such as: qubits, gates, quantum circuits, quantum states, superposition, entanglement, and measurement. Take a refresher on linear algebra concepts if needed, also complexity of algorithms and "big O" notation. Read a comparison of various qubit technologies to understand their trade-offs.

- Watch "Error Mitigation Landscape | QDC 2025" by Eddins, https://youtu.be/ix52wx4_zek?si=cQOlxhhAGXXqhYCe


## ▸ Module 1: Overview

Module 1: Quantum Programming Fundamentals, in the NISQ Era

How can I get started with quantum programming for my application domain? How do I make use of noisy quantum hardware effectively?

Suitable for a research audience of programming practitioners, this module helps attendees understand the fundamental concepts of quantum computing including qubits, superposition, entanglement, quantum gates, and circuits. Participants will construct a personal quantum computing development environment, exercise the basic quantum computing workflow, and recognize performance limitations. In lieu of any further course attendance, they will be able to self-study online examples from their domain and use them in their development environment. A homework exercise will explore error accumulation in deep circuits, supported by office hours.


## ▸ Module 1: Topics

- look at mapping QAOA to hardware topologies Qiskit video
- provenance
- transpilation
- variational algorithms, optimization
- grading
- P and NP
- shots estimation
- IBM ordering
- no cloning
- Bell's Theorem
- ancilla
- physical observables
- kinds of sims, tensor nets
- error profile from real backend
- magic vs. pure states
- Clifford: takes a Pauli and returns a Pauli
- error mitigation - transform bias into variance
- dynamical decoupling
- pauli twirling
- bqp-complete
- example of phase kickback
- statevector vs samples
- algorithm zoo
- clifford, cost
- eigenstate / eigenvalue of gate
- QASM
- role of uncomputation
- traveling salesman
- teleportation
- kinds of error - stochastic
- no cloning
- P and NP
- provenance
- error models and characterization, mitigation techniques
- magic states
- formulating as a Hamiltonian
- max cut
- QFT
- classical vs. quantum data

Bell vs GHZ
\(|\Phi ^{+}\rangle =\frac{|00\rangle +|11\rangle }{\sqrt{2}}\): Both qubits are \(|0\rangle \) or both are \(|1\rangle \).\(|\Psi ^{+}\rangle =\frac{|01\rangle +|10\rangle }{\sqrt{2}}\): On



    + quantum concepts
        - qubit representation, Bloch sphere
        - quantum states and measurement
        - quantum gates and circuits
        - superposition and interference
        - entanglement, Bell states, Bell's theorem
        - no-cloning theorem
        - Hilbert space
        - comparisons to classical computing, other programming paradigms
        - state preparation, amplitude encoding
    + hardware concepts
        - qubits (physical vs. logical, qubit implementations), connectivity, coherence
        - gates and circuits, errors
        - quantum volume and other metrics
    + implementation options
        - Qiskit, Pennylane, CUDA-Q, others
        - Classiq
    + hybrid quantum-classical workflow
        - variational algorithms, issues with
        - design, code, optimize, execute, analyze
        - transpilation and optimization, circuit mapping, coupling maps, gate sets
        - noise models and simulators
        - execution - estimator vs. shot-based, simulator vs. hardware, cloud access




+ Module 1
    + Key takeaways / value proposition
        - Understand the fundamental concepts of quantum computing, including qubits, superposition, entanglement, quantum gates, and circuits.
        - Construct a personal quantum computing development environment, understand the basic quantum computing workflow, and understand performance limitations.
        - In lieu of any further course attendance, be able to self-study online examples from your domain and use them in your development environment.
    + Pre-Work (15 min)
        - From instructions, set up a working Python environment with a quantum SDK (e.g. Qiskit) and simulator.
    + Lecture (1 hour)
        + Quantum fundamentals
            - Qubit representation, Bloch sphere
            - Quantum states and measurement, probabilistic outcomes
            - Quantum gates and circuits, gate sets
            - Alternatives to gate-based models, e.g. annealing
            - Superposition and interference
            - Entanglement, Bell states, Bell's theorem
            - No-cloning theorem
            - Hilbert space
            - Comparisons to classical computing, other programming paradigms
        + Hardware fundamentals
            - Qubits (physical vs. logical, qubit implementations), connectivity, coherence
            - Gates and circuits, errors
            - Quantum volume and other metrics
        + Quantum software fundamentals
            - Quantum programming workflow: design, code, optimize, execute, analyze
            - Quantum SDKs - Qiskit and others
            - Transpilation and optimization, circuit mapping, coupling maps, gate sets
            - Noise models and simulators
            - Execution - estimator vs. shot-based, simulator vs. hardware, cloud access
        + Quantum subroutines
            - Quantum Phase Estimation (QPE)
            - Quantum Fourier Transform
        + Quantum Advantage
            - When and why QC may provide advantage over classical computing
            - Uses: chemistry, cryptography, quantum linear systems, quantum machine learning
            - Criteria for advantage: speed, accuracy, problem types, logical qubits at scale, hybrid connectivity
            - Roadmap to advantage, current state of the art, DARPA metrics
            - Cryptography implications
    + Lab (1 hour)
        - Get a working Python environment with a quantum SDK (e.g. Qiskit) and simulator.
        - Implement a basic "hello world" quantum circuit (e.g. create and measure a Bell state).
        - Understand the capabilities of your laptop to handle a number of qubits on a simulator by constructing a GHZ state of increasing size until performance degrades significantly. Turn the noise model on and off to see the impact.
    + Post-Work: Error Accumulation (2 hours)
        - Write (or use from a lib) a <circuit, e.g. QPE> which takes <some variable problem size> as input and demonstrates how error accumulates with circuit depth on a simulator with a specific noise model. Graph input size vs. error rate.
        - Using a problem size of significance, understand the runtime cost as a function of shots.
        - Attend office hours to discuss any questions on pre-work, lecture, or lab.


## ▸ Module 1: Homework







