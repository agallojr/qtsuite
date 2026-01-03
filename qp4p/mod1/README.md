# Quantum Programming for Programmers (QP4P) - Module 1

## ▸ Module 1: Prerequisites

- Attend or otherwise consume QP4P Module 0. Review the foundational concepts from Module 0 to ensure familiarity with quantum computing basics and Python programming.

- Intermediate-level Python programming skills, including familiarity with functions, classes, libraries such as numpy and matplotlib, and the use of virtual environments or other means to manage dependencies.

- A working Python environment set to the libraries published for the class. We recommend the use of uv (https://docs.astral.sh/uv/). We have published a specification file (see ../pyproject.toml) that defines the required dependencies. Sample code will be provided to verify the environment. Please come to class with a working environment as we will not have time to troubleshoot environment issues during class.

- Run the Module 0 examples. 

- Read "Quantum Computation and Quantum Information" by Mike & Ike, Chapters 1-4, 7, or seek alternative introductory materials to build foundational understanding on topics such as: qubits, gates, quantum circuits, quantum states, superposition, entanglement, and measurement. Take a refresher on linear algebra concepts if needed, also complexity of algorithms and "big O" notation. Read a comparison of various qubit technologies to understand their trade-offs.


## ▸ Module 1: Overview

Module 1: Quantum Programming Fundamentals, in the NISQ Era

How can I get started with quantum programming for my application domain? How do I make use of noisy quantum hardware effectively?

Suitable for a research audience of programming practitioners, this module helps attendees understand the fundamental concepts of quantum computing including qubits, superposition, entanglement, quantum gates, and circuits. Participants will construct a personal quantum computing development environment, exercise the basic quantum computing workflow, recognize performance limitations, and understand the benefits of various error mitigation techniques. In lieu of any further course attendance, they will be able to self-study online examples from their domain and use them in their development environment. A homework exercise will explore error accumulation in deep circuits, supported by office hours.


## ▸ Module 1: Topics

Module 1 will repeat some of the material and many of the examples from Module 0 but with a deeper treatment. Additional material will also be presented. 

+ Introduction and Overview
    - Course overview and objectives: modules 1, 2, 3
    - Utility & advantage: for what, when?
    - Hardware calibration, metadata tracking - experimental repeatability?

+ Quantum programming fundamentals
    - Bells theorem, GHZ states - how big?
    - Observables
    - Pure states vs. magic, state distillation
    - No cloning theorem
    - Uncomputation
    - Classical vs. quantum data
    - Gate types, Pauli, Clifford, annealing
    - Mapping problems to known problems: complexity classes, the Hamiltonian formulation
    - Transpilation, compilation, QASM interop, pulse level programming
    - Major software vendors & platforms, coding & estimation tools, Python vs C++, IBM & NVIDIA
    - Hybrid computing, co-scheduling
    - Kinds of simulators / emulators, cloud offerings

+ Overview of select quantum algorithms
    - Linear systems (code: ax=b): module 2 details CFD use cases
    - Variational Quantum Eigensolver (code: lattice): issues with NISQ variational algorithms
    - Quantum Phase Estimation (code: gs_qpe): module 3 details MAT use cases
    - other: Max Cut, QFT, AOA, QML, TSP
    - formulating as a Hamiltonian
    - Algorithm Zoo

+ <5 min break>

+ Noise & error mitigation
    - Types of noise: depolarizing, amplitude damping, phase damping
    - Error models, model from backend, impact on quantum algorithms
    - Error mitigation techniques: DD, Pauli twirling, ZNE, etc.
    - Code examples to study noise impact
    - Algorithm degeneracy, over-iterating, shots estimation

+ Homework 1

+ Q & A

+ Office Hours


## ▸ Module 1: Homework

We have provided two algorithmic implementations of computing the ground state of a molecule - QPE and VQE. Your task is to compare their performance, accuracy, and practical applicability across different molecular systems and noise conditions, leveraging the parameters of the two algorithms and with the addition of one or more error mitigation techniques.

Analyze the following aspects:
- Computational complexity and scaling
- Required quantum resources (qubits, circuit depth)
- Required classical resources for pre- and post-processing, for simulating the quantum computer
- Sensitivity to noise and utilized error mitigation techniques
- Convergence behavior and optimization strategies
- Accuracy vs. classical reference values
- Practical implementation challenges

Two hard requirements: you must use a shot-based approach on a named backend - one large enough to hold the molecule of interest. 

Try to implement the most complex molecule possible on your available hardware. Try and be creative about the use of such hardware when dealing with computational bottlenecks.

Document your findings and provide recommendations for when to use each approach. Submit your analysis as a short report (2-3 pages) detailing your methodology, results, and conclusions, including any insights about when QPE might be preferred over VQE or vice versa. Include specific examples and quantitative comparisons where possible.

Some suggestions:

- Consider using techniques from this IBM tutorial on error mitigation in the software layer: https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques and/or watch this video "Error Mitigation Landscape | QDC 2025" by Eddins, https://youtu.be/ix52wx4_zek?si=cQOlxhhAGXXqhYCe

- Or maybe you prefer to experiment with some non-Qiskit solutions, or those which can interoperate with Qiskit backends using QASM. For example, you could try interfacing with Cirq or PennyLane and use error mitigation provided by mitiq: https://github.com/unitaryfoundation/mitiq - "your mileage may vary".

- You should be able to use the provided course code to select a backend and then zero out its noise model to simulate a noiseless environment for baseline comparison.

- The sweeper can be used to drive a launcher script to directly compare two algorithms over the same molecular systems and other parameters.

- Take a look at the two supported VQE methods - they do not function the same, and don't take the same set of optimizers. This is a good example of "research grade" software being released even by the major providers maybe before it's fully mature - just be aware there can be bugs and unsupported features, and you may find the authors to be unashamed!

- Be prepared to debug and adapt the code as needed - this is part of the learning experience!




