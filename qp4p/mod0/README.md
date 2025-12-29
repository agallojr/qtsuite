# Quantum Programming for Programmers (QP4P) - Module 0

## Study Guide

Of the players in the emergent quantum computing field, two stand out for their contributions to the educational landscape: superconducting qubit vendor IBM makers of Qiskit the leading quantum programming framework, and to a lesser extent the photonics company Xanadu, makers of the Pennylane SDK. The educational materials from both companies - online documents, tutorials, videos, and sequential courses - are an excellent starting point.

Of particular note are the videos from John Watrous at IBM, especially the 18-part series "Understanding Quantum Computing", https://www.youtube.com/playlist?list=PLOFEBzvs-VvqKKMXX4vbi4EB1uaErFMSO

Individual modules for this course each include their own reading list and prerequisites. The Mike & Ike book has suggested chapter ranges for each module. Additional papers from the arxiv are suggested for each module's main topics.


## Modules

- grading


## Topics

- hype 
- provenance
- real backend error info changes, calibration
- hello world 
- error mitigation
- teleportation
- P and NP
- no cloning
- physical observables
- real backends - IBM, AWS - this class will use sims
- HPC - this class will not use HPC
- magic vs. pure states
- advantage vs. utility
- algorithm zoo
- HPC
- traveling salesman




For an N×N image:

Total pixels = N²
Qubits needed = log₂(N²) = 2·log₂(N)
Column qubits = log₂(N) (half of total)
SWAP gates = floor(log₂(N) / 2)
Examples:

Image Size	Qubits	Column Qubits	SWAP Gates
4×4	4	2	1
8×8	6	3	1
16×16	8	4	2
32×32	10	5	2
64×64	12	6	3
The number of SWAPs grows as O(log N) — very efficient.


i don't undersrand how this woeks with so few swap gates
The key insight is that we're not swapping pixels — we're swapping the meaning of qubit positions.

In amplitude encoding, each basis state |i⟩ corresponds to a pixel index. For a 4×4 image with 4 qubits:

Qubits 0-1 encode the column (0-3)
Qubits 2-3 encode the row (0-3)
A pixel at position (row=2, col=1) is stored in amplitude of state |0110⟩ (row bits: 10, col bits: 01).

Horizontal flip means: pixel at col=1 should move to col=2, col=0→col=3, etc. In binary:

col 0 (00) → col 3 (11)
col 1 (01) → col 2 (10)
col 2 (10) → col 1 (01)
col 3 (11) → col 0 (00)
This is just reversing the bit order of the column qubits. One SWAP between qubit 0 and qubit 1 does exactly that:

|00⟩ → |00⟩
|01⟩ → |10⟩
|10⟩ → |01⟩
|11⟩ → |11⟩
The SWAP gate simultaneously transforms all 16 amplitudes in superposition. That's the quantum parallelism — one gate affects all basis states at once.

For larger images, you need more SWAPs to reverse more column bits, but it's still just reversing bit order, not moving individual pixels.






+ mod0
    - how large a GHZ state can you practically construct on sim? without and with noise?
    - UQ of GHZ state as a function of qubits, t1, t2
    - amplitude encoding, Hilbert space
    - how large an image can you mirror, to some fidelity? with and without noise in the sim? what is the impediment? how could we leverage HPC in this problem? what's the impact of shots?
    - understand what's happening in grover's algorithm - what is the impact of iterations? shots? what's a potential application of grover's?





How does quantum programming differ from classical? Are there applications today, if not, when?

Suitable for a general audience of programming practitioners in the businesses, this overview module introduces the fundamental concepts of quantum computing including qubits, superposition, entanglement, quantum gates, circuits, and how these concepts differ from or are unique compared to classical programming. It covers the basic quantum computing workflow, performance limitations, and key algorithms in quantum computing for searching unsorted data, prime factorization, optimization, and solving linear systems of equations. The module also explores modeling quantum systems using quantum computing techniques and interim hybrid quantum-classical algorithms for quantum chemistry applications. Attendees will gain exposure to the roadmaps of leading quantum hardware and software providers and their projections for quantum utility or advantage in specific applications. A study guide for personal follow-up will be provided, and an optional office hour session will subsequently be made available.


    + executive introduction
    + fundamental differences from classical programming
        - hardware awareness: size, noise, connectivity; current calibration
        - error correction awareness
        - quantum states and measurement
        - probabilistic outcomes
        - Hilbert space
    + similarities to classical programming
        - gates, circuits
        - accelerator model, HPC
        - programming workflow: design, code, optimize, execute, analyze
        - languages, SDKs, cloud access - when native quantum programming abstractions?
        - IDEs: Classiq
    + quantum concepts in code examples
        - qubit
        - superposition
        - entanglement
        - phase kickback
        - algorithms: Grover's, Shor's, QPE, VQE - algorithm zoo
    + quantum advantage & roadmap
        - when and why QC may provide advantage over classical computing
        - uses: chemistry, cryptography, quantum linear systems, quantum machine learning
        - criteria for advantage: speed, accuracy, problem types, logical qubits at scale, hybrid connectivity
        - roadmap to advantage, current state of the art, DARPA metrics
        - cryptography implications
    + QP4P class overview
        - key takeaways / value proposition for each module, topics
        - lab activities
        - pre-work / post-work
        - schedule
        - SME speakers
        - how to sign up



+ Summary Module
    + Key takeaways / value proposition
    + Quantum algorithms overview
    + Lecture
        - tl;dr: safely ignore for now, abstractions will arrive before production utility
        - Feynman, qubit, superposition, interference, entanglement, Hilbert, no cloning, qubit noise, coherence, gate noise
        - comparison to classical computing, other programming paradigms
        - probabilistic outcomes, measurement, experimental perspective
        - qubit types, hardware types, quantum volume
        - when advantage? criteria, roadmap, DARPA
        - uses: chemistry, cryptography, QLSA, QML
        - Criteria for advantage: speed, accuracy, problem types, logical qubits at scale, hybrid connectivity
        - future: higher-level abstractions for scale, AI coder & hybrid accelerator delegation 
        - what criteria should be met to make QC an option?
        - cryptography implications
    + algorithms
        - Grover's, Shor's, Max Cut, VQE, QPE; 
    + tooling
        - workflow: map (design, code), optimize (pipeline), execute, analyze
        - SDKs: Qiskit & other circuit libs, circuit interop, pipeline plugins, sims, cloud
        - estimation
    + applied toys
        - quantum teleportation
        - Ax=b
        - ground state energy of H2
        - estimation of resources




