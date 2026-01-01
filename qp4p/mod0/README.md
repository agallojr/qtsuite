# Quantum Programming for Programmers (QP4P) - Module 0

## Study Guide

Of the players in the emergent quantum computing field, two stand out for their contributions to the educational landscape: superconducting qubit vendor IBM makers of Qiskit the leading quantum programming framework, and to a lesser extent the photonics company Xanadu, makers of the Pennylane SDK. The educational materials from both companies are an excellent starting point - online documents, tutorials, videos on YouTube, and complete sequential courses.

Of particular note are the videos from John Watrous at IBM, especially the 18-part series "Understanding Quantum Computing", https://www.youtube.com/playlist?list=PLOFEBzvs-VvqKKMXX4vbi4EB1uaErFMSO

Individual modules for this course each include their own reading list and prerequisites. The Mike & Ike book has suggested chapter ranges for each module. Additional papers from the arxiv are suggested for each module's main topics.


## Goals of the Course


## Goals of this Module

How does quantum programming differ from classical? Are there applications today, if not, when?

Suitable for a general audience of programming practitioners, this overview module introduces the fundamental concepts of quantum computing including qubits, superposition, entanglement, quantum gates, circuits, and how these concepts differ from or are unique compared to classical programming. It covers the basic quantum computing workflow, performance limitations, and key algorithms in quantum computing for searching unsorted data, prime factorization, optimization, and solving linear systems of equations. The module also explores modeling quantum systems using quantum computing techniques and interim hybrid quantum-classical algorithms for quantum chemistry applications. Attendees will gain exposure to the roadmaps of leading quantum hardware and software providers and their projections for quantum utility or advantage in specific applications. A study guide for personal follow-up will be provided, and an optional office hour session will subsequently be made available.


## Course Modules

- grading


## Module 0: Topics

+ Introduction and Overview
    - Executive context
    - Course overview and objectives: modules 1, 2, 3
    - What is quantum computing? Feynman's vision, probabilistic nature
    - Why study quantum computing? performance & scale beyond classical, Hilbert space
    - tl;dr: safely ignore for now, abstractions will arrive before production utility
    - Utility & advantage: for what, when?
    - Current hardware landscape: hype & reality, types & connections, size, calibration
    - Noise & error mitigation: code examples allow studying impact of noise
    - Current software landscape: maturity, interop, abstractions, circuits & pulses, IDEs
    - IBM & NVIDIA, Python & C++, estimation

+ Quantum basics 
    - qubits vs. bits (code: state preparation & measurement, phase, magic, Bloch sphere)
    - superposition & Schrödinger's cat (code: Hadamard gate)
    - entanglement & the multi-body problem (code: Bell / GHZ states)
    - amplitude encoding (code: amplitude, image mirroring shots study)
    - phase kickback (code: phase_kickback)
    - observables

+ Quantum programming framework
    - Map the problem to a quantum-native and/or quantum-classical formulation
    - Optimize and transpile for the target hardware
    - Execute on quantum hardware or simulator
    - Analyze and post-process results
    - Mapping problems to known problems: complexity classes

+ Hybrid quantum-classical
    - Variational algorithms, pre- & post-processing
    - HPC co-scheduling vs. cloud QPUs, new accelerator(s)
    - workflows & metadata

+ Overview of quantum algorithms
    - Shor's factoring: enterprise IT use cases
    - Grover's search (code: grovers)
    - Linear systems (code: ax=b): module 2 details CFD use cases
    - Variational Quantum Eigensolver (code: lattice): issues with NISQ variational algorithms
    - Quantum Phase Estimation (code: gs_qpe): module 3 details MAT use cases
    - other: QAOA, QML, TSP
    - formulating as a Hamiltonian
    - Algorithm Zoo

+ QP4P class: final plug
    - modules, labs, pre-reqs, schedule, SMEs, sign-up


## Module Library Dependencies

All examples for all modules use the same set of dependent libraries. These include Qiskit and its sub-libraries, numpy, matplotlib, and json for data processing and visualization. A suitable version of Python is also defined.

While the code examples use IBM's Qiskit, we do not require an account with the IBM cloud. All examples can run locally with simulated backends, although the set of backends is limited to those publicly available.

We recommend the use of uv (https://docs.astral.sh/uv/). We have published here a specification file (see ./pyproject.toml) that defines the required dependencies. 


## Running the Sweeps, Gathering Data & Metadata

In the later modules, we show examples of building and executing quantum circuits in more complex algorithmic workflows. The core of these implementations take many parameters and return structured data. Thus workflows can be defined in an input file (TOML format) and swept using the provided qp4p_sweeper tool, with any interactive displays saved for the end of the workflow and later analysis. Examples of running parameter sweeps can be found in the module src directories, e.g. `python mod1/src/ex_image_flip_sweep.py`.

The sweeper tool is located in the qp4p package and can be invoked with various configuration options to run parameter sweeps. One thing it does is keep the output isolated in its own directory for easy organization. Each case in the sweep is also stored in its own separate directory. By separating data generation from post-processing we allow for multiple post-processing steps.

The toml file allows groupings of swept cases and will lay down metadata about the group which can be used by postprocessing scripts to identify the cases within the group and other parameters for the post-processing step. Variables in the toml with a "_" prefix are arguments to the sweeper itself and are not swept (e.g. path to root outdir).

For example, here we run the core image_flip demo with a parameter sweep defined in the toml input file, running just one group of cases defined in that input file:
```bash
python -m qp4p_sweeper mod1/src/image_flip.py mod1/input/image_flip.toml --group size_study
```

The core algorithm json output includes many metadata fields about the problem, solution, and execution environment. Students are encouraged to consider adding their own custom metadata fields to capture domain-specific insights. These can include performance metrics, hardware-specific details, or any other relevant information for their particular use case. Such metadata can be invaluable for tracking experimental results and comparing different approaches. 

Quantum computing by its nature at this time can create many pieces of interim artifacts and results, and they have importance in the current research. Keeping track of these artifacts is essential for comprehensive analysis and reproducibility. Students are encouraged to consider what workflow tools work best for them in this regard.

Additionally, quantum solutions at this time are often hybrid classical-quantum approaches where the classical part can itself be heavyweight and a bottleneck. HPC is useful here. The student is encouraged to consider which workflow tools work best for them in managing these hybrid workloads, such as containerization, with on-prem and cloud orchestration.


## Module 0 Post-Work

If you intend to take module 1, we suggest reviewing these questions to better prepare:

- How large a GHZ state can you practically construct on your machine? without and with noise?

- At the edge of the GHZ state you can simulate, after multiple runs, what can be said about the reliability and consistency of the results? as a function of qubits, t1, t2?

- How large an image can you mirror, to what fidelity? with and without noise in the sim? what is the impediment? how might we leverage classical HPC in this problem? what's the impact of shots?

- Run a study of the lattice model, to understand the impact of size, ansatz, noise, optimizer, and iterations. Experiment with models of different legacy IBM machines (backends) and compare performance.

- See also the documentation for Module 1: mod1/README.md for more pre-requisites.

